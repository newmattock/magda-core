#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <future>
#include <cstring>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "api/clip_api.hpp"
#include "api/magda_api.hpp"
#include "api/project_api.hpp"
#include "api/remote_api.hpp"
#include "api/track_api.hpp"
#include "audio/AudioBridge.hpp"
#include "audio/processors/base/DeviceProcessor.hpp"
#include "audio/plugins/compiled/CompiledPluginRegistry.hpp"
#include "audio/plugins/InternalPluginRegistry.hpp"
#include "core/AutomationInfo.hpp"
#include "core/AutomationManager.hpp"
#include "core/ClipManager.hpp"
#include "core/TrackManager.hpp"
#include "core/DeviceInfo.hpp"
#include "core/MidiNoteCommands.hpp"
#include "core/ParameterInfo.hpp"
#include "core/ParameterUtils.hpp"
#include "core/UndoManager.hpp"
#include "engine/AudioEngine.hpp"
#include "engine/ExternalPluginStateDialect.hpp"
#include "engine/TracktionEngineWrapper.hpp"
#include "project/ProjectInfo.hpp"
#include "project/ProjectManager.hpp"
#include "version.hpp"

namespace {

void printUsage(std::ostream& out);

juce::File fileFromArg(const juce::String& path) {
    if (juce::File::isAbsolutePath(path))
        return juce::File(path);
    return juce::File::getCurrentWorkingDirectory().getChildFile(path);
}

juce::File defaultOutputFor(const juce::File& input) {
    auto parent = input.getParentDirectory();
    auto stem = input.getFileNameWithoutExtension() + "_roundtrip";
    return parent.getChildFile(stem + ".mgd");
}

// The engine of the running headless session. The app registers its engine
// with TrackManager (which is how ParameterInfo display-text providers reach
// the plugins); the CLI deliberately does not, since that also switches on
// hardware MIDI monitoring and routing forwarding. Display text for the dump
// and param_info goes through this pointer instead.
magda::AudioEngine* gHeadlessEngine = nullptr;

class HeadlessEngineSession {
  public:
    ~HeadlessEngineSession() {
        if (gHeadlessEngine == engine_.get())
            gHeadlessEngine = nullptr;
    }

    bool initialize() {
        engine_ = magda::createDefaultAudioEngine({.headless = true});
        if (!engine_->initialize()) {
            error_ = "Failed to initialize MAGDA engine";
            engine_.reset();
            return false;
        }
        gHeadlessEngine = engine_.get();
        return true;
    }

    magda::AudioEngine& engine() {
        return *engine_;
    }

    const juce::String& error() const {
        return error_;
    }

  private:
    std::unique_ptr<magda::AudioEngine> engine_;
    juce::String error_;
};

bool restoreProjectTiming(magda::AudioEngine& engine, const magda::ProjectInfo& info) {
    engine.setTempo(info.tempo);
    engine.setTimeSignature(info.timeSignatureNumerator, info.timeSignatureDenominator);
    return true;
}

std::optional<double> parseDouble(const juce::String& text) {
    const auto input = text.trim().toStdString();
    if (input.empty())
        return std::nullopt;

    double value = 0.0;
    char trailing = 0;
    std::istringstream stream(input);
    if (!(stream >> value) || (stream >> trailing) || !std::isfinite(value))
        return std::nullopt;
    return value;
}

std::optional<int> parseInt(const juce::String& text) {
    const auto input = text.trim().toStdString();
    if (input.empty())
        return std::nullopt;

    int value = 0;
    char trailing = 0;
    std::istringstream stream(input);
    if (!(stream >> value) || (stream >> trailing))
        return std::nullopt;
    return value;
}

std::optional<int> parsePositiveInt(const juce::String& text) {
    auto value = parseInt(text);
    if (value && *value > 0)
        return value;
    return std::nullopt;
}

std::optional<magda::TrackType> parseTrackType(const juce::String& text) {
    auto normalized = text.trim().toLowerCase();
    if (normalized == "audio" || normalized == "midi")
        return magda::TrackType::Audio;
    if (normalized == "group")
        return magda::TrackType::Group;
    if (normalized == "aux")
        return magda::TrackType::Aux;
    if (normalized == "chord")
        return magda::TrackType::Chord;
    return std::nullopt;
}

// Internal instruments that actually produce sound when driven with MIDI
// notes via `add-internal-instrument`. The engine's device model accepts any
// id string and silently renders -120 dBFS for ids that match nothing, which
// turns agent typos into silent projects. This allowlist is the load-time
// contract: anything not on it is rejected with a structured error instead.
// "4osc" (FourOsc) plus the compiled Faust instruments from
// CompiledPluginRegistry (magda_polysynth, magda_fm, the percussion set).
bool isKnownAudibleInstrument(const juce::String& pluginId) {
    const auto normalized = pluginId.trim().toLowerCase();
    if (normalized == "4osc")
        return true;
    if (const auto* spec =
            magda::daw::audio::compiled::findCompiledPluginSpec(normalized))
        return spec->isInstrument;
    return false;
}

juce::String knownInstrumentHint() {
    juce::StringArray names;
    names.add("4osc");
    for (const auto* spec : magda::daw::audio::compiled::getAllCompiledPluginSpecs())
        if (spec->isInstrument)
            names.add(spec->pluginId);
    return names.joinIntoString(", ");
}

// Native (non-compiled) devices an agent may insert as an effect: browser-
// visible, creatable on a track, not an instrument, MIDI generator or meter.
// This is how the MIDI-keyed "sidechain" ducker is reached.
bool isInsertableInternalEffect(const magda::daw::audio::InternalPluginSpec& spec) {
    using magda::daw::audio::InternalPluginCreateMode;
    return spec.pluginId != nullptr && spec.showInBrowser && !spec.isInstrument &&
           spec.canCreateOnTrack && spec.createMode != InternalPluginCreateMode::Unsupported &&
           !magda::daw::audio::internalPluginHasTag(spec, "midi-generator") &&
           !magda::daw::audio::internalPluginHasTag(spec, "analysis");
}

juce::String knownEffectHint() {
    juce::StringArray names;
    for (const auto* spec : magda::daw::audio::compiled::getAllCompiledPluginSpecs())
        if (!spec->isInstrument)
            names.add(spec->pluginId);
    for (const auto* spec : magda::daw::audio::getAllInternalPluginSpecs())
        if (spec != nullptr && isInsertableInternalEffect(*spec) && !names.contains(spec->pluginId))
            names.add(spec->pluginId);
    return names.joinIntoString(", ");
}

// An insert effect the engine can create by id.
struct EffectChoice {
    juce::String pluginId;          // canonical id
    juce::String displayName;
    bool sidechainDevice = false;   // the "sidechain" ducker: main chain only
};

std::optional<EffectChoice> resolveInternalEffect(const juce::String& query) {
    const auto id = query.trim();
    if (id.isEmpty())
        return std::nullopt;
    if (const auto* spec = magda::daw::audio::compiled::findCompiledPluginSpec(id)) {
        if (spec->isInstrument || spec->pluginId == nullptr)
            return std::nullopt;
        return EffectChoice{juce::String(spec->pluginId),
                            juce::String(spec->displayName != nullptr ? spec->displayName
                                                                      : spec->pluginId),
                            false};
    }
    if (const auto* spec = magda::daw::audio::findInternalPluginSpec(id)) {
        if (!isInsertableInternalEffect(*spec))
            return std::nullopt;
        return EffectChoice{juce::String(spec->pluginId),
                            juce::String(spec->displayName != nullptr ? spec->displayName
                                                                      : spec->pluginId),
                            magda::daw::audio::internalPluginHasTag(*spec, "sidechain")};
    }
    return std::nullopt;
}

magda::DeviceInfo makeInternalEffectDevice(const juce::String& pluginId, const juce::String& name) {
    magda::DeviceInfo device;
    device.name = name.isNotEmpty() ? name : pluginId;
    device.manufacturer = "MAGDA";
    device.pluginId = pluginId;
    device.uniqueId = pluginId;
    device.fileOrIdentifier = pluginId;
    device.isInstrument = false;
    device.deviceType = magda::DeviceType::Effect;
    device.format = magda::PluginFormat::Internal;
    return device;
}

double gainToDb(double gain) {
    return gain > 0.0 ? 20.0 * std::log10(gain) : -144.0;
}

double dbToGain(double db) {
    return db <= -144.0 ? 0.0 : std::pow(10.0, db / 20.0);
}

juce::var clipToJson(const magda::ClipInfo& clip) {
    auto value = magda::remote::toJson(magda::remote::makeClipDto(clip));
    if (auto* obj = value.getDynamicObject()) {
        obj->setProperty("loopEnabled", clip.loopEnabled);
        if (clip.loopEnabled)
            obj->setProperty("loopLengthBeats", clip.loopLengthBeats);
    }
    return value;
}

juce::var sendsToJson(const magda::TrackInfo& track) {
    juce::Array<juce::var> sends;
    for (const auto& send : track.sends) {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("to", send.destTrackId);
        obj->setProperty("busIndex", send.busIndex);
        obj->setProperty("level", static_cast<double>(send.level));
        obj->setProperty("levelDb", gainToDb(send.level));
        sends.add(juce::var(obj));
    }
    return sends;
}

juce::var trackToJson(magda::MagdaApi& api, const magda::TrackInfo& track) {
    auto value = magda::remote::toJson(magda::remote::makeTrackDto(track));
    auto* obj = value.getDynamicObject();

    juce::Array<juce::var> clips;
    for (auto clipId : api.clips().getClipsOnTrack(track.id)) {
        if (auto* clip = api.clips().getClip(clipId))
            clips.add(clipToJson(*clip));
    }
    obj->setProperty("clips", clips);
    // Protocol 3: the mixer in the units agents think in, and the sends.
    obj->setProperty("volumeDb", gainToDb(track.volume));
    obj->setProperty("sends", sendsToJson(track));
    if (track.auxBusIndex >= 0)
        obj->setProperty("auxBusIndex", track.auxBusIndex);
    return value;
}

// Mixer state of one channel ("master" for the master channel).
juce::var mixerStateJson(magda::TrackId trackId) {
    auto& trackManager = magda::TrackManager::getInstance();
    auto* obj = new juce::DynamicObject();
    if (trackId == magda::MASTER_TRACK_ID) {
        const auto& master = trackManager.getMasterChannel();
        obj->setProperty("id", "master");
        obj->setProperty("volume", static_cast<double>(master.volume));
        obj->setProperty("volumeDb", gainToDb(master.volume));
        obj->setProperty("pan", static_cast<double>(master.pan));
        obj->setProperty("muted", master.muted);
    } else if (const auto* track = trackManager.getTrack(trackId)) {
        obj->setProperty("id", track->id);
        obj->setProperty("name", track->name);
        obj->setProperty("volume", static_cast<double>(track->volume));
        obj->setProperty("volumeDb", gainToDb(track->volume));
        obj->setProperty("pan", static_cast<double>(track->pan));
        obj->setProperty("muted", track->muted);
        obj->setProperty("soloed", track->soloed);
        obj->setProperty("output", track->audioOutputDevice);
        obj->setProperty("sends", sendsToJson(*track));
    }
    return juce::var(obj);
}

const char* curveName(magda::AutomationCurveType type) {
    switch (type) {
        case magda::AutomationCurveType::Linear:
            return "linear";
        case magda::AutomationCurveType::Bezier:
            return "bezier";
        case magda::AutomationCurveType::Step:
            return "step";
        case magda::AutomationCurveType::HardCorner:
            return "corner";
    }
    return "linear";
}

juce::var automationTargetJson(const magda::AutomationTarget& target) {
    using Kind = magda::ControlTarget::Kind;
    auto* obj = new juce::DynamicObject();
    switch (target.kind) {
        case Kind::TrackVolume:
            obj->setProperty("kind", "volume");
            break;
        case Kind::TrackPan:
            obj->setProperty("kind", "pan");
            break;
        case Kind::SendLevel:
            obj->setProperty("kind", "send");
            break;
        case Kind::PluginParam:
            obj->setProperty("kind", "param");
            break;
        case Kind::DeviceMacro:
            obj->setProperty("kind", "macro");
            break;
        case Kind::ModParam:
            obj->setProperty("kind", "mod");
            break;
        case Kind::Tempo:
            obj->setProperty("kind", "tempo");
            break;
    }
    if (!target.isEditScoped()) {
        const auto trackId = target.devicePath.trackId;
        if (trackId == magda::MASTER_TRACK_ID)
            obj->setProperty("trackId", "master");
        else
            obj->setProperty("trackId", trackId);
    }
    if (target.kind == Kind::SendLevel) {
        obj->setProperty("busIndex", target.sendBusIndex);
        if (const auto* track =
                magda::TrackManager::getInstance().getTrack(target.devicePath.trackId))
            for (const auto& send : track->sends)
                if (send.busIndex == target.sendBusIndex)
                    obj->setProperty("to", send.destTrackId);
    }
    if (target.kind == Kind::PluginParam || target.kind == Kind::DeviceMacro) {
        obj->setProperty("deviceId", target.devicePath.getDeviceId());
        obj->setProperty("paramIndex", target.paramIndex);
    }
    return juce::var(obj);
}

// Every automation lane with its points: normalized lane value plus the
// value in the target's real units (dB for volume/sends, -1..1 for pan).
juce::var automationToJson() {
    juce::Array<juce::var> lanes;
    for (const auto& lane : magda::AutomationManager::getInstance().getLanes()) {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("laneId", lane.id);
        obj->setProperty("target", automationTargetJson(lane.target));
        obj->setProperty("name", lane.getDisplayName());
        obj->setProperty("type", lane.isAbsolute() ? "absolute" : "clip-based");
        if (lane.isAbsolute()) {
            const auto info = magda::getParameterInfoForTarget(lane.target);
            juce::Array<juce::var> points;
            for (const auto& point : lane.absolutePoints) {
                auto* p = new juce::DynamicObject();
                p->setProperty("beat", point.beatPosition);
                p->setProperty("value", point.value);
                p->setProperty("real", static_cast<double>(magda::ParameterUtils::normalizedToReal(
                                           static_cast<float>(point.value), info)));
                p->setProperty("curve", curveName(point.curveType));
                points.add(juce::var(p));
            }
            obj->setProperty("points", points);
            if (info.unit.isNotEmpty())
                obj->setProperty("unit", info.unit);
        }
        lanes.add(juce::var(obj));
    }
    return lanes;
}

// ---------------------------------------------------------------------------
// Parameter meaning (serve protocol 2): every parameter in `dump` carries its
// display text, default, scale and -- for discrete parameters -- its labels
// and the value each label sits at, so agents can say "Hall" instead of 1.0.
// Hosted plugins get their engine-wrapper Dry/Wet pair back, flagged
// owner:"host", instead of it silently shadowing the plugin's own controls.
// ---------------------------------------------------------------------------
const char* parameterScaleName(magda::ParameterScale scale) {
    switch (scale) {
        case magda::ParameterScale::Linear:
            return "linear";
        case magda::ParameterScale::Logarithmic:
            return "log";
        case magda::ParameterScale::Exponential:
            return "exp";
        case magda::ParameterScale::Discrete:
            return "discrete";
        case magda::ParameterScale::Boolean:
            return "boolean";
        case magda::ParameterScale::FaderDB:
            return "fader-db";
    }
    return "linear";
}

// Native value of choice `i` of `n`: internal discrete parameters store the
// index itself; anything else spreads the choices across its range.
double choiceNativeValue(const magda::ParameterInfo& p, size_t i, size_t n) {
    const double lo = p.minValue;
    const double hi = p.maxValue;
    if (n <= 1)
        return lo;
    if (std::abs((hi - lo) - static_cast<double>(n - 1)) < 1e-6)
        return lo + static_cast<double>(i);
    return lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
}

// Display text of `value` (real units) for a parameter of the device at `path`:
// the device's own processor first (hosted plugins answer with their own text,
// e.g. "LP 12 dB" or "523 Hz"), then MAGDA's generic formatter.
juce::String parameterDisplayText(const magda::ChainNodePath& path, const magda::ParameterInfo& p,
                                  float value) {
    if (p.displayText != nullptr && gHeadlessEngine != nullptr && path.isValid()) {
        if (auto* bridge = gHeadlessEngine->getAudioBridge()) {
            if (auto* processor = bridge->getDeviceProcessor(path)) {
                // Processors format TE-native values; project when the model's
                // display range differs from TE's (as ParameterUtils does).
                float native = value;
                const float teSpan = p.teMaxValue - p.teMinValue;
                const bool sameRange = std::abs(p.minValue - p.teMinValue) < 1e-6f &&
                                       std::abs(p.maxValue - p.teMaxValue) < 1e-6f;
                if (!sameRange && teSpan > 0.0f)
                    native = p.teMinValue +
                             magda::ParameterUtils::realToNormalized(value, p) * teSpan;
                auto text = processor->formatParameterValue(p.paramIndex, native).trim();
                if (text.isNotEmpty())
                    return text;
            }
        }
    }
    return magda::ParameterUtils::formatValue(value, p).trim();
}

void addParameterMeaning(juce::DynamicObject& obj, const magda::ChainNodePath& path,
                         const magda::ParameterInfo& p) {
    obj.setProperty("default", static_cast<double>(p.defaultValue));
    obj.setProperty("scale", parameterScaleName(p.scale));
    if (!p.choices.empty()) {
        juce::Array<juce::var> labels;
        juce::Array<juce::var> values;
        for (size_t i = 0; i < p.choices.size(); ++i) {
            labels.add(p.choices[i]);
            values.add(choiceNativeValue(p, i, p.choices.size()));
        }
        obj.setProperty("choices", labels);
        obj.setProperty("choiceValues", values);
        obj.setProperty("steps", static_cast<int>(p.choices.size()));
    } else if (p.scale == magda::ParameterScale::Boolean) {
        obj.setProperty("steps", 2);
    }
    const auto text = parameterDisplayText(path, p, p.currentValue);
    if (text.isNotEmpty())
        obj.setProperty("text", text);
}

juce::var parameterToJson(const magda::ChainNodePath& path, const magda::ParameterInfo& p,
                          bool hostWrapper) {
    auto* obj = new juce::DynamicObject();
    obj->setProperty("index", p.paramIndex);
    obj->setProperty("name", p.name);
    if (p.stableId.isNotEmpty())
        obj->setProperty("stableId", p.stableId);
    obj->setProperty("value", static_cast<double>(p.currentValue));
    obj->setProperty("min", static_cast<double>(p.minValue));
    obj->setProperty("max", static_cast<double>(p.maxValue));
    if (p.unit.isNotEmpty())
        obj->setProperty("unit", p.unit);
    if (hostWrapper) {
        obj->setProperty("owner", "host");
        obj->setProperty("wrapperRole", p.wrapperRole == magda::WrapperRole::DryGain   ? "dry"
                                        : p.wrapperRole == magda::WrapperRole::WetGain ? "wet"
                                                                                        : "other");
    }
    addParameterMeaning(*obj, path, p);
    return juce::var(obj);
}

const magda::DeviceInfo* findModelDevice(int deviceId, magda::ChainNodePath* outPath = nullptr) {
    auto& trackManager = magda::TrackManager::getInstance();
    const auto path = trackManager.findDevicePath(deviceId);
    if (!path.isValid())
        return nullptr;
    if (outPath != nullptr)
        *outPath = path;
    return trackManager.getDeviceInChainByPath(path);
}

// Replace a device's parameter list in the dump with the meaning-rich view
// (wrapper pair first, as Tracktion orders them) and mark it as saved state.
void enrichDeviceJson(juce::var& deviceJson) {
    auto* obj = deviceJson.getDynamicObject();
    if (obj == nullptr)
        return;
    magda::ChainNodePath path;
    const auto* info = findModelDevice(static_cast<int>(obj->getProperty("id")), &path);
    if (info == nullptr)
        return;
    juce::Array<juce::var> parameters;
    for (const auto& wrapper : info->wrapperParameters)
        parameters.add(parameterToJson(path, wrapper, true));
    for (const auto& param : info->parameters)
        parameters.add(parameterToJson(path, param, false));
    obj->setProperty("parameters", parameters);
    obj->setProperty("pluginId", info->pluginId);
    obj->setProperty("persistent", true);
    obj->setProperty("loadState", info->loadState == magda::DeviceLoadState::Loading ? "loading"
                                  : info->loadState == magda::DeviceLoadState::Failed ? "failed"
                                                                                      : "loaded");
    // Protocol 3: where the device sits (main chain before the fader, or the
    // post-fader list) and what keys it.
    obj->setProperty("position", path.isPostFx() ? "post" : "pre");
    if (info->sidechain.isActive()) {
        auto* sidechain = new juce::DynamicObject();
        sidechain->setProperty("type", info->sidechain.type == magda::SidechainConfig::Type::MIDI
                                           ? "midi"
                                           : "audio");
        sidechain->setProperty("source", info->sidechain.sourceTrackId);
        obj->setProperty("sidechain", juce::var(sidechain));
    }
}

juce::String dumpProjectJson(magda::MagdaApi& api) {
    const auto& project = api.project().getCurrentProjectInfo();
    auto value = magda::remote::toJson(magda::remote::makeProjectDto(project));
    auto* root = value.getDynamicObject();

    juce::Array<juce::var> tracks;
    for (const auto& track : api.tracks().getTracks())
        tracks.add(trackToJson(api, track));
    root->setProperty("tracks", tracks);

    // Device graph incl. post-fx devices and their parameter metadata, so
    // external agents can introspect (and set) parameters without guessing.
    const auto graph =
        magda::remote::makeDeviceGraphDto(api.tracks().getTracks());
    juce::Array<juce::var> devices;
    for (const auto& device : graph.devices) {
        auto json = magda::remote::toJson(device);
        enrichDeviceJson(json);
        devices.add(json);
    }

    // Protocol 3: the master channel (its mixer state and insert devices,
    // listed with trackId -2) and every automation lane.
    if (const auto* masterTrack = magda::TrackManager::getInstance().getTrack(magda::MASTER_TRACK_ID)) {
        const std::vector<magda::TrackInfo> masterOnly{*masterTrack};
        juce::Array<juce::var> masterDeviceIds;
        for (const auto& device : magda::remote::makeDeviceGraphDto(masterOnly).devices) {
            auto json = magda::remote::toJson(device);
            enrichDeviceJson(json);
            masterDeviceIds.add(device.id);
            devices.add(json);
        }
        auto master = mixerStateJson(magda::MASTER_TRACK_ID);
        if (auto* masterObj = master.getDynamicObject())
            masterObj->setProperty("devices", masterDeviceIds);
        root->setProperty("master", master);
    }
    root->setProperty("devices", devices);
    root->setProperty("automation", automationToJson());

    return juce::JSON::toString(value, true);
}

struct CommandResult {
    bool ok = true;
    juce::String error;
};

// Free-function twin of CommandDispatcher::fail for code outside the class
// (the serve loop builds CommandResults directly).
CommandResult makeFail(const juce::String& error) {
    return {false, error};
}

// External AU/VST3 hosting lives on the Tracktion wrapper; the CLI only holds
// the generic AudioEngine&, so downcast once per use.
magda::TracktionEngineWrapper* asTracktionWrapper(magda::AudioEngine& engine) {
    return dynamic_cast<magda::TracktionEngineWrapper*>(&engine);
}

// Resolve a scanned external plugin by exact createIdentifierString match
// first, then by the FIRST case-insensitive name match.
std::optional<juce::PluginDescription> findExternalPluginDescription(
    magda::AudioEngine& engine, const juce::String& query) {
    const auto needle = query.trim();
    if (needle.isEmpty())
        return std::nullopt;

    std::optional<juce::PluginDescription> byName;
    for (const auto& candidate : engine.getKnownPluginTypes()) {
        if (candidate.createIdentifierString() == needle)
            return candidate;
        if (!byName && candidate.name.equalsIgnoreCase(needle))
            byName = candidate;
    }
    return byName;
}

// ============================================================================
// Serve protocol 3 model helpers, shared by the exec commands and the JSON
// ops: track references, sends, sidechain keys, insert effects, clip
// placement and automation.
// ============================================================================

bool isJsonNumber(const juce::var& v) {
    return v.isInt() || v.isInt64() || v.isDouble();
}

std::optional<int> intProperty(const juce::var& request, const char* key) {
    const auto v = request.getProperty(key, juce::var());
    if (isJsonNumber(v))
        return static_cast<int>(static_cast<double>(v));
    return std::nullopt;
}

std::optional<double> doubleProperty(const juce::var& request, const char* key) {
    const auto v = request.getProperty(key, juce::var());
    if (isJsonNumber(v)) {
        const auto value = static_cast<double>(v);
        if (std::isfinite(value))
            return value;
    }
    return std::nullopt;
}

// "master" (or -2) names the master channel; anything else is a track id.
std::optional<magda::TrackId> parseTrackRef(const juce::String& text) {
    const auto token = text.trim().toLowerCase();
    if (token == "master" || token == "-2")
        return magda::MASTER_TRACK_ID;
    if (const auto value = parseInt(token); value && *value >= 0)
        return static_cast<magda::TrackId>(*value);
    return std::nullopt;
}

std::optional<magda::TrackId> trackProperty(const juce::var& request, const char* key) {
    const auto v = request.getProperty(key, juce::var());
    if (isJsonNumber(v)) {
        const auto value = static_cast<int>(static_cast<double>(v));
        if (value == magda::MASTER_TRACK_ID || value >= 0)
            return static_cast<magda::TrackId>(value);
        return std::nullopt;
    }
    if (v.isString())
        return parseTrackRef(v.toString());
    return std::nullopt;
}

std::optional<bool> parseSwitch(const juce::String& text) {
    const auto token = text.trim().toLowerCase();
    if (token == "on" || token == "true" || token == "1" || token == "yes")
        return true;
    if (token == "off" || token == "false" || token == "0" || token == "no")
        return false;
    return std::nullopt;
}

std::optional<bool> boolProperty(const juce::var& request, const char* key) {
    const auto v = request.getProperty(key, juce::var());
    if (v.isBool() || isJsonNumber(v))
        return static_cast<bool>(v);
    if (v.isString())
        return parseSwitch(v.toString());
    return std::nullopt;
}

juce::String trackLabel(magda::TrackId trackId) {
    return trackId == magda::MASTER_TRACK_ID ? juce::String("master") : juce::String(trackId);
}

// Aux and group buses (and the master) carry no clips.
bool trackTakesClips(const magda::TrackInfo& track) {
    return track.type != magda::TrackType::Aux && track.type != magda::TrackType::Group &&
           track.type != magda::TrackType::Master;
}

// Where a track's audio goes besides the master: sends, output routing, group.
std::vector<magda::TrackId> downstreamOf(const magda::TrackInfo& track) {
    std::vector<magda::TrackId> out;
    for (const auto& send : track.sends)
        out.push_back(send.destTrackId);
    if (track.audioOutputDevice.startsWith("track:"))
        out.push_back(
            track.audioOutputDevice.fromFirstOccurrenceOf("track:", false, false).getIntValue());
    if (track.parentId != magda::INVALID_TRACK_ID)
        out.push_back(track.parentId);
    return out;
}

// True when audio from `from` reaches `target` through sends, routing or groups.
bool feedsInto(magda::TrackId from, magda::TrackId target) {
    auto& trackManager = magda::TrackManager::getInstance();
    std::vector<magda::TrackId> pending{from};
    std::set<magda::TrackId> seen;
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (id == target)
            return true;
        if (id == magda::MASTER_TRACK_ID || !seen.insert(id).second)
            continue;
        if (const auto* track = trackManager.getTrack(id))
            for (const auto next : downstreamOf(*track))
                pending.push_back(next);
    }
    return false;
}

std::optional<int> findSendBus(magda::TrackId source, magda::TrackId dest) {
    auto& trackManager = magda::TrackManager::getInstance();
    if (source == magda::MASTER_TRACK_ID || dest == magda::MASTER_TRACK_ID)
        return std::nullopt;
    const auto* src = trackManager.getTrack(source);
    const auto* dst = trackManager.getTrack(dest);
    if (src == nullptr || dst == nullptr)
        return std::nullopt;
    for (const auto& send : src->sends)
        if (send.destTrackId == dest ||
            (dst->auxBusIndex >= 0 && send.busIndex == dst->auxBusIndex))
            return send.busIndex;
    return std::nullopt;
}

// Adds (or reuses) the post-fader send from `source` to `dest` and sets its
// level (linear gain, 1.0 = 0 dB); returns the send's bus index.
std::optional<int> ensureSend(magda::TrackId source, magda::TrackId dest,
                              std::optional<double> level, juce::String& error) {
    auto& trackManager = magda::TrackManager::getInstance();
    if (source == magda::MASTER_TRACK_ID || trackManager.getTrack(source) == nullptr) {
        error = "unknown source track " + trackLabel(source);
        return std::nullopt;
    }
    if (dest == magda::MASTER_TRACK_ID || trackManager.getTrack(dest) == nullptr) {
        error = "unknown destination track " + trackLabel(dest) +
                " (send to an aux or group track; every track already feeds the master)";
        return std::nullopt;
    }
    if (source == dest) {
        error = "a track cannot send to itself";
        return std::nullopt;
    }
    if (level && (!std::isfinite(*level) || *level < 0.0 || *level > 2.0)) {
        error = "send level must be a linear gain in [0, 2] (-inf..+6 dB)";
        return std::nullopt;
    }
    auto bus = findSendBus(source, dest);
    if (!bus) {
        if (feedsInto(dest, source)) {
            error = "track " + juce::String(dest) + " already feeds track " + juce::String(source) +
                    "; this send would make a feedback loop";
            return std::nullopt;
        }
        trackManager.addSend(source, dest);
        bus = findSendBus(source, dest);
        if (!bus) {
            error = "the engine refused the send (a track has at most 8 sends)";
            return std::nullopt;
        }
    }
    if (level)
        trackManager.setSendLevel(source, *bus, static_cast<float>(*level));
    return bus;
}

// Keys a device from a source track -- MIDI notes retrigger the "sidechain"
// ducker's envelope, audio feeds a compressor's detector -- or clears the key
// (source INVALID_TRACK_ID). `applied` reports the route that was set.
bool applySidechain(magda::AudioEngine& engine, int deviceId, magda::TrackId source,
                    std::optional<magda::SidechainConfig::Type> requested, juce::String& error,
                    juce::String& applied) {
    auto& trackManager = magda::TrackManager::getInstance();
    magda::ChainNodePath path;
    const auto* device = findModelDevice(deviceId, &path);
    if (device == nullptr) {
        error = "unknown device " + juce::String(deviceId);
        return false;
    }
    if (source == magda::INVALID_TRACK_ID) {
        trackManager.clearSidechain(deviceId);
        applied = "none";
        return true;
    }
    if (path.isPostFx()) {
        error = "device " + juce::String(deviceId) +
                " sits after the fader; sidechain keys only reach devices in the main "
                "(pre-fader) chain -- add it with position \"pre\"";
        return false;
    }
    if (source == magda::MASTER_TRACK_ID || trackManager.getTrack(source) == nullptr) {
        error = "unknown sidechain source track " + trackLabel(source);
        return false;
    }
    if (source == path.trackId) {
        error = "a device cannot be keyed from its own track";
        return false;
    }
    const bool ducker = magda::daw::audio::internalPluginHasTag(device->pluginId, "sidechain");
    const auto type = requested.value_or(ducker ? magda::SidechainConfig::Type::MIDI
                                                : magda::SidechainConfig::Type::Audio);
    if (type == magda::SidechainConfig::Type::Audio) {
        auto* bridge = engine.getAudioBridge();
        auto plugin = bridge != nullptr ? bridge->getPlugin(path) : nullptr;
        if (plugin == nullptr || !plugin->canSidechain()) {
            error = device->name + " has no audio sidechain input" +
                    juce::String(ducker ? " (the sidechain ducker is keyed by notes: use type midi)"
                                        : " (magda_compressor has one)");
            return false;
        }
    } else if (!ducker && device->mods.empty()) {
        error = device->name +
                " has no modulators for MIDI notes to retrigger; use the \"sidechain\" ducker "
                "(add_fx plugin sidechain) or type audio";
        return false;
    }
    trackManager.setSidechainSource(deviceId, source, type);
    const auto* after = findModelDevice(deviceId);
    if (after == nullptr || after->sidechain.sourceTrackId != source ||
        after->sidechain.type != type) {
        error = "the engine did not accept the sidechain route";
        return false;
    }
    applied = type == magda::SidechainConfig::Type::MIDI ? "midi" : "audio";
    return true;
}

// Inserts a built-in effect: in the main chain ("pre", before the fader,
// where automation, modulation and sidechain keys reach it) or in the
// post-fader list.
magda::DeviceId addInternalEffect(magda::TrackId trackId, const EffectChoice& effect,
                                  const juce::String& name, bool preFader, juce::String& error) {
    auto& trackManager = magda::TrackManager::getInstance();
    if (trackManager.getTrack(trackId) == nullptr) {
        error = "unknown track " + trackLabel(trackId);
        return magda::INVALID_DEVICE_ID;
    }
    if (effect.sidechainDevice && !preFader) {
        error = effect.pluginId + " only works in the main (pre-fader) chain";
        return magda::INVALID_DEVICE_ID;
    }
    const auto device = makeInternalEffectDevice(effect.pluginId, name);
    const auto deviceId = preFader ? trackManager.addDeviceToTrack(trackId, device)
                                   : trackManager.addDeviceToPostFx(trackId, device);
    if (deviceId == magda::INVALID_DEVICE_ID)
        error = "track " + trackLabel(trackId) + " cannot host " + effect.pluginId;
    return deviceId;
}

// ---- clip placement ---------------------------------------------------------

struct ClipSpan {
    magda::ClipId id = magda::INVALID_CLIP_ID;
    double start = 0.0;
    double end = 0.0;
    juce::String name;
};

std::vector<ClipSpan> overlappingClips(magda::TrackId trackId, double start, double length,
                                       magda::ClipId ignore) {
    std::vector<ClipSpan> hits;
    auto& clipManager = magda::ClipManager::getInstance();
    constexpr double epsilon = 1.0e-9;
    const double end = start + length;
    for (const auto clipId : clipManager.getClipsOnTrack(trackId)) {
        if (clipId == ignore)
            continue;
        const auto* clip = clipManager.getClip(clipId);
        if (clip == nullptr || clip->view != magda::ClipView::Arrangement)
            continue;
        const double clipStart = clip->placement.startBeat;
        const double clipEnd = clipStart + clip->placement.lengthBeats;
        if (clipStart < end - epsilon && clipEnd > start + epsilon)
            hits.push_back({clipId, clipStart, clipEnd, clip->name});
    }
    return hits;
}

juce::String describeClipSpans(const std::vector<ClipSpan>& spans) {
    juce::StringArray parts;
    for (const auto& span : spans)
        parts.add("clip " + juce::String(span.id) + " \"" + span.name + "\" [" +
                  juce::String(span.start) + ", " + juce::String(span.end) + ")");
    return parts.joinIntoString(", ");
}

// First start at or after `desired` where a clip of `length` fits.
double nextFreeStart(magda::TrackId trackId, double desired, double length, magda::ClipId ignore) {
    double candidate = desired;
    for (int guard = 0; guard < 100000; ++guard) {
        const auto hits = overlappingClips(trackId, candidate, length, ignore);
        if (hits.empty())
            break;
        for (const auto& hit : hits)
            candidate = std::max(candidate, hit.end);
    }
    return candidate;
}

double lastClipEnd(magda::TrackId trackId) {
    double end = 0.0;
    auto& clipManager = magda::ClipManager::getInstance();
    for (const auto clipId : clipManager.getClipsOnTrack(trackId))
        if (const auto* clip = clipManager.getClip(clipId))
            end = std::max(end, clip->placement.startBeat + clip->placement.lengthBeats);
    return end;
}

const magda::ClipInfo* arrangementClip(magda::ClipId clipId, juce::String& error) {
    const auto* clip = magda::ClipManager::getInstance().getClip(clipId);
    if (clip == nullptr || clip->view != magda::ClipView::Arrangement) {
        error = "unknown arrangement clip " + juce::String(clipId);
        return nullptr;
    }
    return clip;
}

bool clipTrackUsable(magda::TrackId trackId, juce::String& error) {
    const auto* track = magda::TrackManager::getInstance().getTrack(trackId);
    if (trackId == magda::MASTER_TRACK_ID || track == nullptr || !trackTakesClips(*track)) {
        error = "track " + trackLabel(trackId) + " cannot hold clips (aux/group buses and the "
                "master take audio from other tracks)";
        return false;
    }
    return true;
}

// Moves a clip to (trackId, start). A clip changing track is parked past every
// clip on both tracks first, so the engine's last-write-wins overlap handling
// can only ever touch the destination range.
bool moveClipChecked(magda::ClipId clipId, std::optional<magda::TrackId> track, double start,
                     bool replace, juce::String& error) {
    auto& clipManager = magda::ClipManager::getInstance();
    const auto* clip = arrangementClip(clipId, error);
    if (clip == nullptr)
        return false;
    if (!std::isfinite(start) || start < 0.0) {
        error = "start must be a beat >= 0";
        return false;
    }
    const auto fromTrack = clip->trackId;
    const auto destination = track.value_or(fromTrack);
    if (!clipTrackUsable(destination, error))
        return false;
    const double length = clip->placement.lengthBeats;
    if (!replace) {
        const auto hits = overlappingClips(destination, start, length, clipId);
        if (!hits.empty()) {
            error = "the clip would overlap " + describeClipSpans(hits) +
                    " (overlap replace trims what is underneath)";
            return false;
        }
    }
    if (destination != fromTrack) {
        const double parking = std::max(lastClipEnd(fromTrack), lastClipEnd(destination)) + 64.0;
        clipManager.moveClipBeats(clipId, parking);
        clipManager.moveClipToTrack(clipId, destination);
    }
    clipManager.moveClipBeats(clipId, start);
    return true;
}

bool resizeClipChecked(magda::ClipId clipId, double length, bool replace, juce::String& error) {
    const auto* clip = arrangementClip(clipId, error);
    if (clip == nullptr)
        return false;
    if (!std::isfinite(length) || length <= 0.0) {
        error = "length must be > 0 beats";
        return false;
    }
    if (!replace) {
        const auto hits = overlappingClips(clip->trackId, clip->placement.startBeat, length, clipId);
        if (!hits.empty()) {
            error = "the clip would overlap " + describeClipSpans(hits) +
                    " (overlap replace trims what is underneath)";
            return false;
        }
    }
    magda::ClipManager::getInstance().resizeClipBeats(clipId, length);
    return true;
}

// overlap: "error" | "replace" (the copy trims what is underneath) | "shift"
// (the copy moves later until it fits).
magda::ClipId duplicateClipChecked(magda::ClipId clipId, std::optional<magda::TrackId> track,
                                   double start, const juce::String& overlap, juce::String& error) {
    const auto* clip = arrangementClip(clipId, error);
    if (clip == nullptr)
        return magda::INVALID_CLIP_ID;
    if (!std::isfinite(start) || start < 0.0) {
        error = "start must be a beat >= 0";
        return magda::INVALID_CLIP_ID;
    }
    const auto destination = track.value_or(clip->trackId);
    if (!clipTrackUsable(destination, error))
        return magda::INVALID_CLIP_ID;
    const double length = clip->placement.lengthBeats;
    double at = start;
    if (overlap == "shift") {
        at = nextFreeStart(destination, start, length, magda::INVALID_CLIP_ID);
    } else if (overlap != "replace") {
        const auto hits = overlappingClips(destination, start, length, magda::INVALID_CLIP_ID);
        if (!hits.empty()) {
            error = "the copy would overlap " + describeClipSpans(hits) +
                    " (overlap replace trims what is underneath, shift moves the copy later)";
            return magda::INVALID_CLIP_ID;
        }
    }
    const auto copy = magda::ClipManager::getInstance().duplicateClipAtBeats(clipId, at, destination);
    if (copy == magda::INVALID_CLIP_ID)
        error = "the engine could not duplicate clip " + juce::String(clipId);
    return copy;
}

// Loops a MIDI clip's first `loopLength` beats across `totalLength` beats of
// timeline (one bar of drums written once, played for sixteen).
bool setClipLoopChecked(magda::ClipId clipId, bool enabled, std::optional<double> loopLength,
                        std::optional<double> totalLength, bool replace, juce::String& error) {
    auto& clipManager = magda::ClipManager::getInstance();
    const auto* clip = arrangementClip(clipId, error);
    if (clip == nullptr)
        return false;
    if (!clip->isMidi()) {
        error = "only MIDI clips loop here";
        return false;
    }
    const double bpm = magda::ProjectManager::getInstance().getCurrentProjectInfo().tempo;
    if (!enabled) {
        clipManager.setClipLoopEnabled(clipId, false, bpm);
        return true;
    }
    const double loop = loopLength.value_or(clip->placement.lengthBeats);
    if (!std::isfinite(loop) || loop <= 0.0) {
        error = "loopLength must be > 0 beats";
        return false;
    }
    if (totalLength) {
        if (!std::isfinite(*totalLength) || *totalLength <= 0.0) {
            error = "length must be > 0 beats";
            return false;
        }
        if (!replace) {
            const auto hits =
                overlappingClips(clip->trackId, clip->placement.startBeat, *totalLength, clipId);
            if (!hits.empty()) {
                error = "the looped clip would overlap " + describeClipSpans(hits) +
                        " (overlap replace trims what is underneath)";
                return false;
            }
        }
    }
    clipManager.setMidiLoopLengthBeats(clipId, loop, bpm);
    clipManager.setClipLoopEnabled(clipId, true, bpm);
    if (totalLength)
        clipManager.resizeClipBeats(clipId, *totalLength);
    return true;
}

// Parses [{start, length, pitch, velocity?}] (clip-relative beats) into notes
// that must fit inside a clip `clipLength` beats long.
bool parseClipNotes(const juce::var& notesVar, double clipLength,
                    std::vector<magda::MidiNote>& notes, juce::String& error) {
    if (!notesVar.isArray()) {
        error = "notes must be an array";
        return false;
    }
    notes.reserve(static_cast<size_t>(notesVar.size()));
    for (const auto& n : *notesVar.getArray()) {
        const auto start = n.getProperty("start", juce::var());
        const auto length = n.getProperty("length", juce::var());
        const auto pitch = n.getProperty("pitch", juce::var());
        const auto velocity = n.getProperty("velocity", 100);
        if (!isJsonNumber(start) || !isJsonNumber(length) || !isJsonNumber(pitch) ||
            !isJsonNumber(velocity)) {
            error = "each note needs numeric start, length, pitch (and velocity)";
            return false;
        }
        magda::MidiNote note;
        note.startBeat = static_cast<double>(start);
        note.lengthBeats = static_cast<double>(length);
        note.noteNumber = static_cast<int>(static_cast<double>(pitch));
        note.velocity = juce::jlimit(0, 127, static_cast<int>(static_cast<double>(velocity)));
        if (!std::isfinite(note.startBeat) || note.startBeat < 0.0 ||
            !std::isfinite(note.lengthBeats) || note.lengthBeats <= 0.0 || note.noteNumber < 0 ||
            note.noteNumber > 127) {
            error = "invalid note (start >= 0, length > 0, pitch 0..127)";
            return false;
        }
        if (note.startBeat + note.lengthBeats > clipLength + 1.0e-6) {
            error = "note at " + juce::String(note.startBeat) + " (+" +
                    juce::String(note.lengthBeats) + ") runs past the clip end (" +
                    juce::String(clipLength) + " beats; starts are clip-relative)";
            return false;
        }
        notes.push_back(note);
    }
    return true;
}

size_t addNotesToClip(magda::ClipId clipId, std::vector<magda::MidiNote> notes) {
    if (notes.empty())
        return 0;
    magda::ClipManager::BatchScope notifications;
    auto command =
        std::make_unique<magda::AddMultipleMidiNotesCommand>(clipId, std::move(notes), "Add MIDI Notes");
    auto* raw = command.get();
    magda::UndoManager::getInstance().executeCommand(std::move(command));
    return raw->getInsertedIndices().size();
}

// ---- automation -----------------------------------------------------------

const magda::ParameterInfo* resolveDeviceParameter(const magda::DeviceInfo& device,
                                                   const juce::var& ref, bool& isHost) {
    isHost = false;
    auto byIndex = [&](int index) -> const magda::ParameterInfo* {
        for (const auto& p : device.parameters)
            if (p.paramIndex == index)
                return &p;
        for (const auto& p : device.wrapperParameters)
            if (p.paramIndex == index) {
                isHost = true;
                return &p;
            }
        return nullptr;
    };
    if (isJsonNumber(ref))
        return byIndex(static_cast<int>(static_cast<double>(ref)));
    auto name = ref.toString().trim();
    if (const auto index = parseInt(name))
        return byIndex(*index);
    const bool hostOnly = name.startsWithIgnoreCase("host:");
    if (hostOnly)
        name = name.substring(5).trim();
    if (!hostOnly)
        for (const auto& p : device.parameters)
            if (p.stableId.equalsIgnoreCase(name) || p.name.equalsIgnoreCase(name))
                return &p;
    for (const auto& p : device.wrapperParameters)
        if (p.name.equalsIgnoreCase(name)) {
            isHost = true;
            return &p;
        }
    return nullptr;
}


struct AutomationTargetSpec {
    magda::AutomationTarget target;
    magda::ParameterInfo info;   // real range/scale the lane values are normalised against
    bool gainTarget = false;     // volume/sends: values in dB (or linear with unit "gain")
    bool discrete = false;       // labelled/stepped parameter: points step by default
};

// {kind:"volume"|"pan", trackId} | {kind:"send", trackId, to|busIndex} |
// {kind:"param", deviceId, param:name|index}
std::optional<AutomationTargetSpec> resolveAutomationTarget(const juce::var& spec,
                                                            juce::String& error) {
    if (!spec.isObject()) {
        error = "automation requires a target object {kind: volume|pan|send|param, ...}";
        return std::nullopt;
    }
    auto& trackManager = magda::TrackManager::getInstance();
    const auto kind = spec.getProperty("kind", "").toString().trim().toLowerCase();
    AutomationTargetSpec out;
    if (kind == "volume" || kind == "pan" || kind == "send") {
        const auto trackId = trackProperty(spec, "trackId");
        if (!trackId || trackManager.getTrack(*trackId) == nullptr) {
            error = kind + " automation requires a known trackId (or \"master\")";
            return std::nullopt;
        }
        if (kind == "volume") {
            out.target = magda::ControlTarget::trackVolume(*trackId);
            out.gainTarget = true;
        } else if (kind == "pan") {
            out.target = magda::ControlTarget::trackPan(*trackId);
        } else {
            std::optional<int> bus;
            if (const auto to = trackProperty(spec, "to"))
                bus = findSendBus(*trackId, *to);
            else if (const auto index = intProperty(spec, "busIndex"))
                bus = *index;
            if (!bus) {
                error = "track " + trackLabel(*trackId) +
                        " has no such send; add it first (op send / add-send)";
                return std::nullopt;
            }
            out.target = magda::ControlTarget::sendLevel(*trackId, *bus);
            out.gainTarget = true;
        }
    } else if (kind == "param") {
        const auto deviceId = intProperty(spec, "deviceId");
        magda::ChainNodePath path;
        const auto* device = deviceId ? findModelDevice(*deviceId, &path) : nullptr;
        if (device == nullptr) {
            error = "param automation requires a known deviceId";
            return std::nullopt;
        }
        if (path.isPostFx()) {
            error = "device " + juce::String(*deviceId) +
                    " sits after the fader, where the engine cannot automate it; re-add it with "
                    "position \"pre\"";
            return std::nullopt;
        }
        bool isHost = false;
        const auto ref = spec.getProperty("param", juce::var());
        const auto* parameter = resolveDeviceParameter(*device, ref, isHost);
        if (parameter == nullptr) {
            error = "unknown parameter " + ref.toString() + " on device " + juce::String(*deviceId);
            return std::nullopt;
        }
        if (isHost) {
            error = "the engine wrapper's Dry/Wet gains cannot be automated; automate the "
                    "plugin's own mix control instead";
            return std::nullopt;
        }
        out.target = magda::ControlTarget::pluginParam(path, parameter->paramIndex);
        out.discrete = !parameter->choices.empty() ||
                       parameter->scale == magda::ParameterScale::Discrete ||
                       parameter->scale == magda::ParameterScale::Boolean;
    } else {
        error = "target kind must be volume, pan, send or param";
        return std::nullopt;
    }
    out.info = magda::getParameterInfoForTarget(out.target);
    return out;
}

std::optional<magda::AutomationCurveType> parseCurve(const juce::String& text) {
    const auto token = text.trim().toLowerCase();
    if (token == "linear" || token == "ramp")
        return magda::AutomationCurveType::Linear;
    if (token == "step" || token == "hold")
        return magda::AutomationCurveType::Step;
    if (token == "smooth" || token == "bezier")
        return magda::AutomationCurveType::Bezier;
    if (token == "corner")
        return magda::AutomationCurveType::HardCorner;
    return std::nullopt;
}

// A point value in the target's real units -> the lane's 0..1 value.
std::optional<double> normaliseAutomationValue(const AutomationTargetSpec& spec,
                                               const juce::var& value, bool linearGain,
                                               juce::String& error) {
    double real = 0.0;
    if (value.isString()) {
        const auto label = value.toString().trim();
        const auto& choices = spec.info.choices;
        bool found = false;
        for (size_t i = 0; i < choices.size() && !found; ++i) {
            if (choices[i].equalsIgnoreCase(label)) {
                real = choiceNativeValue(spec.info, i, choices.size());
                found = true;
            }
        }
        if (!found) {
            error = "\"" + label + "\" is not one of this parameter's labels" +
                    (choices.empty() ? juce::String(" (it declares none: pass a number)")
                                     : juce::String());
            return std::nullopt;
        }
    } else if (isJsonNumber(value)) {
        real = static_cast<double>(value);
        if (!std::isfinite(real)) {
            error = "automation values must be finite";
            return std::nullopt;
        }
        if (spec.gainTarget && linearGain)
            real = gainToDb(real);
    } else {
        error = "each point needs a numeric value (or a label for labelled parameters)";
        return std::nullopt;
    }
    const double lo = spec.info.minValue;
    const double hi = spec.info.maxValue;
    if (spec.gainTarget) {
        real = juce::jlimit(lo, hi, real);  // below the fader's floor is silence
    } else if (real < lo - 1.0e-6 || real > hi + 1.0e-6) {
        error = "value " + juce::String(real) + " is outside [" + juce::String(lo) + ", " +
                juce::String(hi) + "]";
        return std::nullopt;
    }
    return juce::jlimit(0.0, 1.0,
                        static_cast<double>(magda::ParameterUtils::realToNormalized(
                            static_cast<float>(real), spec.info)));
}

// Writes (or clears) one automation lane:
//   {target:{...}, points:[{beat, value, curve?, tension?}], mode, unit?, normalized?}
// mode: replace (the whole lane), add (merge), range (replace only between
// the first and last new point) or clear (delete the lane: back to the fader
// or knob value). Values are in the target's real units -- dB for volume and
// sends (unit "gain" for linear), -1..1 for pan, the parameter's own range or
// a label for params. Returns an error message, or empty and fills `result`.
juce::String applyAutomationRequest(const juce::var& request, juce::DynamicObject& result) {
    juce::String error;
    const auto spec = resolveAutomationTarget(request.getProperty("target", juce::var()), error);
    if (!spec)
        return error;
    auto& automation = magda::AutomationManager::getInstance();
    const auto mode = request.getProperty("mode", "replace").toString().trim().toLowerCase();
    if (mode != "replace" && mode != "add" && mode != "range" && mode != "clear")
        return "automation mode must be replace, add, range or clear";
    result.setProperty("target", automationTargetJson(spec->target));
    if (mode == "clear") {
        const auto laneId = automation.getLaneForTarget(spec->target);
        if (laneId != magda::INVALID_AUTOMATION_LANE_ID)
            automation.deleteLane(laneId);
        result.setProperty("cleared", laneId != magda::INVALID_AUTOMATION_LANE_ID);
        return {};
    }
    const auto pointsVar = request.getProperty("points", juce::var());
    if (!pointsVar.isArray() || pointsVar.size() == 0)
        return "automation requires a non-empty points array";
    const bool normalized = static_cast<bool>(request.getProperty("normalized", false));
    const bool linearGain = request.getProperty("unit", "").toString().equalsIgnoreCase("gain");
    std::vector<magda::AutomationPoint> points;
    points.reserve(static_cast<size_t>(pointsVar.size()));
    double firstBeat = std::numeric_limits<double>::max();
    double lastBeat = 0.0;
    for (const auto& p : *pointsVar.getArray()) {
        const auto beat = p.getProperty("beat", juce::var());
        if (!isJsonNumber(beat) || !std::isfinite(static_cast<double>(beat)) ||
            static_cast<double>(beat) < 0.0)
            return "each point needs a beat >= 0";
        magda::AutomationPoint point;
        point.beatPosition = static_cast<double>(beat);
        const auto value = p.getProperty("value", juce::var());
        if (normalized) {
            if (!isJsonNumber(value) || !(static_cast<double>(value) >= 0.0) ||
                static_cast<double>(value) > 1.0)
                return "normalized values must be numbers in [0, 1]";
            point.value = static_cast<double>(value);
        } else {
            const auto laneValue = normaliseAutomationValue(*spec, value, linearGain, error);
            if (!laneValue)
                return error;
            point.value = *laneValue;
        }
        const auto curveText = p.getProperty("curve", "").toString();
        if (curveText.isNotEmpty()) {
            const auto curve = parseCurve(curveText);
            if (!curve)
                return "curve must be linear, step or smooth";
            point.curveType = *curve;
        } else {
            point.curveType = spec->discrete ? magda::AutomationCurveType::Step
                                             : magda::AutomationCurveType::Linear;
        }
        const auto tension = p.getProperty("tension", juce::var());
        if (isJsonNumber(tension))
            point.tension = juce::jlimit(-1.0, 1.0, static_cast<double>(tension));
        firstBeat = std::min(firstBeat, point.beatPosition);
        lastBeat = std::max(lastBeat, point.beatPosition);
        points.push_back(point);
    }
    const auto laneId =
        automation.getOrCreateLane(spec->target, magda::AutomationLaneType::Absolute);
    const auto* lane = laneId != magda::INVALID_AUTOMATION_LANE_ID ? automation.getLane(laneId)
                                                                   : nullptr;
    if (lane == nullptr)
        return "the engine cannot automate this target";
    if (!lane->isAbsolute())
        return "this target already has a clip-based automation lane; clear it first (mode clear)";
    if (mode == "add") {
        auto merged = lane->absolutePoints;
        merged.insert(merged.end(), points.begin(), points.end());
        automation.replaceLanePoints(laneId, merged);
    } else if (mode == "range") {
        static_cast<void>(automation.replacePointsInRange(laneId, firstBeat, lastBeat, points));
    } else {
        automation.replaceLanePoints(laneId, points);
    }
    const auto* after = automation.getLane(laneId);
    result.setProperty("laneId", laneId);
    result.setProperty("points",
                       after != nullptr ? static_cast<int>(after->absolutePoints.size()) : 0);
    auto* range = new juce::DynamicObject();
    range->setProperty("min", static_cast<double>(spec->info.minValue));
    range->setProperty("max", static_cast<double>(spec->info.maxValue));
    if (spec->info.unit.isNotEmpty())
        range->setProperty("unit", spec->info.unit);
    result.setProperty("range", juce::var(range));
    return {};
}


class CommandDispatcher {
  public:
    explicit CommandDispatcher(magda::AudioEngine& engine) : engine_(engine) {}

    struct CommandSpec {
        const char* name;
        const char* usage;
        CommandResult (CommandDispatcher::*handler)(const juce::StringArray&, size_t&);
        bool showInUsage = true;
        const char* summary = "";
    };

    static const std::vector<CommandSpec>& commandSpecs() {
        static const std::vector<CommandSpec> specs = {
            {"set-tempo", "set-tempo <bpm>", &CommandDispatcher::setTempo, true,
             "Set the project tempo."},
            {"add-track", "add-track <audio|group|aux|chord> [name]", &CommandDispatcher::addTrack,
             true,
             "Create a track: audio (instrument/MIDI), group (bus its children feed), aux "
             "(return bus that sends feed) or chord."},
            {"set-track-name", "set-track-name <track-id> <name...>",
             &CommandDispatcher::setTrackNameCommand, true, "Rename a track."},
            {"add-internal-instrument", "add-internal-instrument <track-id> <plugin-id> [name]",
             &CommandDispatcher::addInternalInstrument, true,
             "Load a built-in instrument (4osc, magda_polysynth, magda_kick, ...)."},
            {"add-fx", "add-fx <track-id|master> <plugin-id> [name] [--pre|--post]",
             &CommandDispatcher::addFx, true,
             "Insert a built-in effect; post-fader unless --pre (the main chain, where "
             "automation and sidechain keys reach it). The sidechain ducker is always pre."},
            {"set-track-volume", "set-track-volume <track-id|master> <volume 0.0-2.0>",
             &CommandDispatcher::setTrackVolume, true,
             "Set a channel fader as linear gain (1.0 = 0 dB, 2.0 = +6 dB)."},
            {"set-master-volume", "set-master-volume <volume 0.0-2.0>",
             &CommandDispatcher::setMasterVolumeCommand, true,
             "Set the master fader as linear gain."},
            {"set-track-pan", "set-track-pan <track-id|master> <pan -1.0..1.0>",
             &CommandDispatcher::setTrackPanCommand, true,
             "Pan a channel (-1 left, 0 centre, 1 right)."},
            {"set-track-mute", "set-track-mute <track-id|master> <on|off>",
             &CommandDispatcher::setTrackMuteCommand, true, "Mute or unmute a channel."},
            {"set-track-solo", "set-track-solo <track-id> <on|off>",
             &CommandDispatcher::setTrackSoloCommand, true, "Solo or unsolo a track."},
            {"add-send", "add-send <source-id> <dest-id> [level 0.0-2.0]",
             &CommandDispatcher::addSendCommand, true,
             "Send a track post-fader to an aux or group track (level: linear gain, default 1.0)."},
            {"set-send-level", "set-send-level <source-id> <dest-id> <level 0.0-2.0>",
             &CommandDispatcher::setSendLevelCommand, true, "Change a send level (linear gain)."},
            {"remove-send", "remove-send <source-id> <dest-id>",
             &CommandDispatcher::removeSendCommand, true, "Remove a send."},
            {"set-sidechain", "set-sidechain <device-id> <source-track-id|none> [audio|midi]",
             &CommandDispatcher::setSidechainCommand, true,
             "Key a pre-fader device from another track: audio into magda_compressor's detector, "
             "or MIDI notes into the sidechain ducker."},
            {"set-device-parameter", "set-device-parameter <device-id> <param-index> <value>",
             &CommandDispatcher::setDeviceParameter, true,
             "Set a device parameter in its native range (by index, name or stable id)."},
            {"delete-track", "delete-track <track-id>", &CommandDispatcher::deleteTrack, true,
             "Delete a track."},
            {"set-track-input", "set-track-input <track-id> <audio|midi> <track:N|device|all|none>",
             &CommandDispatcher::setTrackInput, true, "Route audio or MIDI input into a track."},
            {"group-track", "group-track <child-id> <group-id>", &CommandDispatcher::groupTrack,
             true, "Put a track in a group: its output then feeds the group bus."},
            {"ungroup-track", "ungroup-track <child-id>", &CommandDispatcher::ungroupTrack, true,
             "Take a track out of its group."},
            {"route-midi-to", "route-midi-to <source-id> <dest-id>",
             &CommandDispatcher::routeMidiTo, true, "Feed one track's MIDI into another track."},
            {"add-midi-clip",
             "add-midi-clip <track-id> <start-beats> <length-beats> [--strict|--replace]",
             &CommandDispatcher::addMidiClip, true,
             "Create a MIDI clip at an absolute beat. Clips on one track never stack: by default "
             "an overlapping clip is pushed later; --strict fails instead, --replace trims what "
             "is underneath."},
            {"add-clip", "add-midi-clip <track-id> <start-beats> <length-beats>",
             &CommandDispatcher::addMidiClip, false, ""},
            {"move-clip", "move-clip <clip-id> <start-beats> [--track <id>] [--replace]",
             &CommandDispatcher::moveClipCommand, true,
             "Move a clip in time and/or to another track (fails on overlap unless --replace)."},
            {"resize-clip", "resize-clip <clip-id> <length-beats> [--replace]",
             &CommandDispatcher::resizeClipCommand, true, "Change a clip's length."},
            {"duplicate-clip", "duplicate-clip <clip-id> <start-beats> [--track <id>] [--replace]",
             &CommandDispatcher::duplicateClipCommand, true,
             "Copy a clip (with its notes) to another position or track."},
            {"rename-clip", "rename-clip <clip-id> <name...>", &CommandDispatcher::renameClipCommand,
             true, "Rename a clip."},
            {"set-clip-loop",
             "set-clip-loop <clip-id> <on|off> [loop-length-beats] [--length <total-beats>]",
             &CommandDispatcher::setClipLoopCommand, true,
             "Loop a MIDI clip's first loop-length beats across its (new) total length."},
            {"delete-clip", "delete-clip <clip-id>", &CommandDispatcher::deleteClip, true,
             "Delete a clip."},
            {"add-midi-note",
             "add-midi-note <clip-id> <start-beats> <note> <length-beats> "
             "[velocity]",
             &CommandDispatcher::addMidiNote, true,
             "Add a note (start relative to the clip). Consecutive notes for one clip are "
             "batched."},
            {"quantize-notes",
             "quantize-notes <clip-id> <grid-beats> <start|length|both> <all|note-index...>",
             &CommandDispatcher::quantizeNotes, true, "Quantize notes to a grid."},
            {"slice-notes", "slice-notes <clip-id> <subdivisions> <all|note-index...>",
             &CommandDispatcher::sliceNotes, true, "Split notes into equal parts."},
            {"transpose-midi-clip", "transpose-midi-clip <clip-id> <semitones>",
             &CommandDispatcher::transposeMidiClip, true, "Transpose every note in a clip."},
            {"add-automation-point",
             "add-automation-point <volume:T|pan:T|send:T:D|param:DEV:PARAM> <beat> <value> "
             "[linear|step|smooth]",
             &CommandDispatcher::addAutomationPoint, true,
             "Add an automation point in real units: dB for volume and sends, -1..1 for pan, the "
             "parameter's own range (or a label) for params. T may be master."},
            {"clear-automation", "clear-automation <volume:T|pan:T|send:T:D|param:DEV:PARAM>",
             &CommandDispatcher::clearAutomation, true,
             "Delete an automation lane (the control returns to its set value)."},
            {"plugins-list", "plugins-list", &CommandDispatcher::pluginsList, true,
             "List scanned external plugins."},
            {"scan-plugins", "scan-plugins", &CommandDispatcher::scanPlugins, true,
             "Scan for external plugins (slow, first run)."},
            {"add-plugin",
             "add-plugin <track-id> <plugin-id-or-name> [--instrument] [--out path]",
             &CommandDispatcher::addPlugin, true,
             "Host an external plugin beside the project model (legacy; serve op add_plugin "
             "saves it with the project)."},
            {"remove-device", "remove-device <ext-id>", &CommandDispatcher::removeDevice, true,
             "Remove a legacy hosted plugin."},
            {"ext-parameters", "ext-parameters <ext-id>", &CommandDispatcher::extParameters, true,
             "List a legacy hosted plugin's parameters."},
            {"set-ext-parameter",
             "set-ext-parameter <ext-id> <param-name> <normalised 0..1>",
             &CommandDispatcher::setExtParameter, true,
             "Set a legacy hosted plugin's parameter."},
            {"dump", "dump --json", &CommandDispatcher::dump, true, "Print the project as JSON."},
        };
        return specs;
    }

    CommandResult execute(const juce::StringArray& tokens, size_t& index) {
        if (index >= static_cast<size_t>(tokens.size()))
            return {};

        const auto command = tokens[static_cast<int>(index++)];
        for (const auto& spec : commandSpecs())
            if (command == spec.name)
                return (this->*spec.handler)(tokens, index);

        return fail("Unknown command: " + command);
    }

    void dumpJson() {
        std::cout << dumpProjectJson(engine_.getMagdaApi()) << "\n";
    }

  private:
    CommandResult setTempo(const juce::StringArray& tokens, size_t& index) {
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("set-tempo requires <bpm>");
        auto bpm = parseDouble(tokens[static_cast<int>(index++)]);
        if (!bpm || *bpm <= 0.0)
            return fail("set-tempo requires a positive numeric bpm");

        engine_.getMagdaApi().project().setTempo(*bpm);
        return {};
    }

    CommandResult addTrack(const juce::StringArray& tokens, size_t& index) {
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("add-track requires <audio|group|aux|chord>");

        const auto typeToken = tokens[static_cast<int>(index++)];
        auto type = parseTrackType(typeToken);
        if (!type)
            return fail("Unsupported track type: " + typeToken);

        juce::String name = juce::String(magda::getTrackTypeName(*type));
        if (index < static_cast<size_t>(tokens.size()) &&
            !isCommand(tokens[static_cast<int>(index)]))
            name = tokens[static_cast<int>(index++)];

        auto id = engine_.getMagdaApi().tracks().createTrack(name, *type);
        std::cout << "track " << id << "\n";
        return {};
    }

    CommandResult deleteTrack(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeInt(tokens, index, "delete-track requires <track-id>");
        if (!trackId)
            return fail(lastParseError_);
        engine_.getMagdaApi().tracks().deleteTrack(*trackId);
        return {};
    }

    CommandResult setTrackInput(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeInt(tokens, index, "set-track-input requires <track-id>");
        if (!trackId)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("set-track-input requires <audio|midi>");

        const auto kind = tokens[static_cast<int>(index++)].trim().toLowerCase();
        if (kind != "audio" && kind != "midi")
            return fail("set-track-input kind must be audio or midi");
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("set-track-input requires <track:N|device|all|none>");

        auto value = tokens[static_cast<int>(index++)];
        if (value == "none")
            value = juce::String();

        // TrackApi has no input-routing setters yet, so reach TrackManager
        // directly (the same singleton path the CLI uses for ProjectManager).
        auto& trackManager = magda::TrackManager::getInstance();
        if (kind == "audio")
            trackManager.setTrackAudioInput(*trackId, value);
        else
            trackManager.setTrackMidiInput(*trackId, value);

        const auto* track = engine_.getMagdaApi().tracks().getTrack(*trackId);
        if (track == nullptr)
            return fail("set-track-input: unknown track " + juce::String(*trackId));

        const auto& applied = kind == "audio" ? track->audioInputDevice : track->midiInputDevice;
        if (applied != value)
            return fail("set-track-input rejected: track " + juce::String(*trackId) + " " + kind +
                        " input is \"" + applied + "\"");

        std::cout << "input " << *trackId << " " << kind << " "
                  << (value.isEmpty() ? juce::String("none") : value) << "\n";
        return {};
    }

    CommandResult groupTrack(const juce::StringArray& tokens, size_t& index) {
        auto childId = takeInt(tokens, index, "group-track requires <child-id>");
        if (!childId)
            return fail(lastParseError_);
        auto groupId = takeInt(tokens, index, "group-track requires <group-id>");
        if (!groupId)
            return fail(lastParseError_);

        // TrackApi has no grouping setters yet, so reach TrackManager directly
        // (same singleton path as set-track-input).
        auto& trackManager = magda::TrackManager::getInstance();
        if (trackManager.getTrack(*childId) == nullptr)
            return fail("group-track: unknown track " + juce::String(*childId));
        if (trackManager.getTrack(*groupId) == nullptr)
            return fail("group-track: unknown track " + juce::String(*groupId));

        trackManager.addTrackToGroup(*childId, *groupId);

        const auto* child = trackManager.getTrack(*childId);
        if (child == nullptr || child->parentId != *groupId)
            return fail("group-track rejected: track " + juce::String(*childId) +
                        " is not a child of track " + juce::String(*groupId));

        std::cout << "group " << *childId << " " << *groupId << "\n";
        return {};
    }

    CommandResult ungroupTrack(const juce::StringArray& tokens, size_t& index) {
        auto childId = takeInt(tokens, index, "ungroup-track requires <child-id>");
        if (!childId)
            return fail(lastParseError_);

        auto& trackManager = magda::TrackManager::getInstance();
        if (trackManager.getTrack(*childId) == nullptr)
            return fail("ungroup-track: unknown track " + juce::String(*childId));

        trackManager.removeTrackFromGroup(*childId);
        std::cout << "ungroup " << *childId << "\n";
        return {};
    }

    CommandResult routeMidiTo(const juce::StringArray& tokens, size_t& index) {
        auto sourceId = takeInt(tokens, index, "route-midi-to requires <source-id>");
        if (!sourceId)
            return fail(lastParseError_);
        auto destId = takeInt(tokens, index, "route-midi-to requires <dest-id>");
        if (!destId)
            return fail(lastParseError_);

        auto& trackManager = magda::TrackManager::getInstance();
        trackManager.routeMidiOutputToTrack(*sourceId, *destId);

        const auto* dest = trackManager.getTrack(*destId);
        if (dest == nullptr)
            return fail("route-midi-to: unknown track " + juce::String(*destId));

        // The model rejects self / unknown / cycle silently, so verify the
        // routing actually landed (mirrors the set-track-input convention).
        const auto expected = "track:" + juce::String(*sourceId);
        if (dest->midiInputDevice != expected)
            return fail("route-midi-to rejected: track " + juce::String(*destId) +
                        " midi input is \"" + dest->midiInputDevice + "\"");

        std::cout << "midi-to " << *sourceId << " " << *destId << "\n";
        return {};
    }

    CommandResult addInternalInstrument(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeInt(tokens, index, "add-internal-instrument requires <track-id>");
        if (!trackId)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("add-internal-instrument requires <plugin-id>");

        const auto pluginId = tokens[static_cast<int>(index++)];
        if (!isKnownAudibleInstrument(pluginId))
            return fail("Unknown internal instrument \"" + pluginId +
                        "\" (it would load but render silence). Known audible "
                        "instruments: " + knownInstrumentHint());
        auto name = pluginId;
        if (index < static_cast<size_t>(tokens.size()) &&
            !isCommand(tokens[static_cast<int>(index)]))
            name = tokens[static_cast<int>(index++)];

        magda::DeviceInfo device;
        device.name = name;
        device.manufacturer = "MAGDA";
        device.pluginId = pluginId;
        device.uniqueId = pluginId;
        device.fileOrIdentifier = pluginId;
        device.isInstrument = true;
        device.deviceType = magda::DeviceType::Instrument;
        device.format = magda::PluginFormat::Internal;

        const auto deviceId = engine_.getMagdaApi().tracks().addDeviceToTrack(*trackId, device);
        if (deviceId == magda::INVALID_DEVICE_ID)
            return fail("Failed to add internal instrument");
        std::cout << "device " << deviceId << "\n";
        return {};
    }

    CommandResult setTrackVolume(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeTrack(tokens, index, "set-track-volume requires <track-id|master>", true);
        if (!trackId)
            return fail(lastParseError_);
        auto volume = takeDouble(tokens, index, "set-track-volume requires <volume>");
        if (!volume)
            return fail(lastParseError_);
        if (*volume < 0.0 || *volume > 2.0)
            return fail("set-track-volume requires a volume in [0.0, 2.0]");

        // TrackManager routes the master to its channel setter.
        auto& trackManager = magda::TrackManager::getInstance();
        trackManager.setTrackVolume(*trackId, static_cast<float>(*volume));
        const double applied = *trackId == magda::MASTER_TRACK_ID
                                   ? trackManager.getMasterChannel().volume
                                   : trackManager.getTrack(*trackId)->volume;
        std::cout << "volume " << trackLabel(*trackId) << " " << applied << "\n";
        return {};
    }

    CommandResult setMasterVolumeCommand(const juce::StringArray& tokens, size_t& index) {
        auto volume = takeDouble(tokens, index, "set-master-volume requires <volume>");
        if (!volume)
            return fail(lastParseError_);
        if (*volume < 0.0 || *volume > 2.0)
            return fail("set-master-volume requires a volume in [0.0, 2.0]");
        magda::TrackManager::getInstance().setMasterVolume(static_cast<float>(*volume));
        std::cout << "volume master " << *volume << "\n";
        return {};
    }

    CommandResult setTrackPanCommand(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeTrack(tokens, index, "set-track-pan requires <track-id|master>", true);
        if (!trackId)
            return fail(lastParseError_);
        auto pan = takeDouble(tokens, index, "set-track-pan requires <pan -1.0..1.0>");
        if (!pan)
            return fail(lastParseError_);
        if (*pan < -1.0 || *pan > 1.0)
            return fail("set-track-pan requires a pan in [-1.0, 1.0]");
        magda::TrackManager::getInstance().setTrackPan(*trackId, static_cast<float>(*pan));
        std::cout << "pan " << trackLabel(*trackId) << " " << *pan << "\n";
        return {};
    }

    CommandResult setTrackMuteCommand(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeTrack(tokens, index, "set-track-mute requires <track-id|master>", true);
        if (!trackId)
            return fail(lastParseError_);
        auto muted = takeSwitch(tokens, index, "set-track-mute requires <on|off>");
        if (!muted)
            return fail(lastParseError_);
        magda::TrackManager::getInstance().setTrackMuted(*trackId, *muted);
        std::cout << "mute " << trackLabel(*trackId) << " " << (*muted ? "on" : "off") << "\n";
        return {};
    }

    CommandResult setTrackSoloCommand(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeTrack(tokens, index, "set-track-solo requires <track-id>", false);
        if (!trackId)
            return fail(lastParseError_);
        auto soloed = takeSwitch(tokens, index, "set-track-solo requires <on|off>");
        if (!soloed)
            return fail(lastParseError_);
        magda::TrackManager::getInstance().setTrackSoloed(*trackId, *soloed);
        std::cout << "solo " << *trackId << " " << (*soloed ? "on" : "off") << "\n";
        return {};
    }

    CommandResult setTrackNameCommand(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeTrack(tokens, index, "set-track-name requires <track-id>", false);
        if (!trackId)
            return fail(lastParseError_);
        const auto name = takeWords(tokens, index);
        if (name.isEmpty())
            return fail("set-track-name requires <name>");
        magda::TrackManager::getInstance().setTrackName(*trackId, name);
        std::cout << "name " << *trackId << " " << name << "\n";
        return {};
    }

    CommandResult addSendCommand(const juce::StringArray& tokens, size_t& index) {
        auto source = takeTrack(tokens, index, "add-send requires <source-id>", false);
        if (!source)
            return fail(lastParseError_);
        auto dest = takeTrack(tokens, index, "add-send requires <dest-id>", false);
        if (!dest)
            return fail(lastParseError_);
        std::optional<double> level;
        if (index < static_cast<size_t>(tokens.size()) &&
            !isCommand(tokens[static_cast<int>(index)])) {
            level = parseDouble(tokens[static_cast<int>(index)]);
            if (!level)
                return fail("add-send level must be a linear gain in [0, 2]");
            ++index;
        }
        juce::String error;
        const auto bus = ensureSend(*source, *dest, level, error);
        if (!bus)
            return fail("add-send: " + error);
        std::cout << "send " << *source << " " << *dest << " bus " << *bus << "\n";
        return {};
    }

    CommandResult setSendLevelCommand(const juce::StringArray& tokens, size_t& index) {
        auto source = takeTrack(tokens, index, "set-send-level requires <source-id>", false);
        if (!source)
            return fail(lastParseError_);
        auto dest = takeTrack(tokens, index, "set-send-level requires <dest-id>", false);
        if (!dest)
            return fail(lastParseError_);
        auto level = takeDouble(tokens, index, "set-send-level requires <level>");
        if (!level)
            return fail(lastParseError_);
        if (!findSendBus(*source, *dest))
            return fail("set-send-level: track " + juce::String(*source) +
                        " has no send to track " + juce::String(*dest));
        juce::String error;
        if (!ensureSend(*source, *dest, level, error))
            return fail("set-send-level: " + error);
        std::cout << "send-level " << *source << " " << *dest << " " << *level << "\n";
        return {};
    }

    CommandResult removeSendCommand(const juce::StringArray& tokens, size_t& index) {
        auto source = takeTrack(tokens, index, "remove-send requires <source-id>", false);
        if (!source)
            return fail(lastParseError_);
        auto dest = takeTrack(tokens, index, "remove-send requires <dest-id>", false);
        if (!dest)
            return fail(lastParseError_);
        const auto bus = findSendBus(*source, *dest);
        if (!bus)
            return fail("remove-send: track " + juce::String(*source) + " has no send to track " +
                        juce::String(*dest));
        magda::TrackManager::getInstance().removeSend(*source, *bus);
        std::cout << "removed-send " << *source << " " << *dest << "\n";
        return {};
    }

    CommandResult setSidechainCommand(const juce::StringArray& tokens, size_t& index) {
        auto deviceId = takeInt(tokens, index, "set-sidechain requires <device-id>");
        if (!deviceId)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("set-sidechain requires <source-track-id|none>");
        const auto sourceToken = tokens[static_cast<int>(index++)].trim().toLowerCase();
        magda::TrackId source = magda::INVALID_TRACK_ID;
        if (sourceToken != "none") {
            const auto parsed = parseTrackRef(sourceToken);
            if (!parsed)
                return fail("set-sidechain source must be a track id or none");
            source = *parsed;
        }
        std::optional<magda::SidechainConfig::Type> type;
        if (index < static_cast<size_t>(tokens.size())) {
            const auto token = tokens[static_cast<int>(index)].trim().toLowerCase();
            if (token == "audio" || token == "midi") {
                type = token == "midi" ? magda::SidechainConfig::Type::MIDI
                                       : magda::SidechainConfig::Type::Audio;
                ++index;
            }
        }
        juce::String error, applied;
        if (!applySidechain(engine_, *deviceId, source, type, error, applied))
            return fail("set-sidechain: " + error);
        std::cout << "sidechain " << *deviceId << " " << applied << " "
                  << (source == magda::INVALID_TRACK_ID ? juce::String("none")
                                                        : juce::String(source))
                  << "\n";
        return {};
    }

    CommandResult addFx(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeTrack(tokens, index, "add-fx requires <track-id|master>", true);
        if (!trackId)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("add-fx requires <plugin-id>");

        const auto pluginId = tokens[static_cast<int>(index++)];
        // Compiled effects plus insertable native devices (the sidechain
        // ducker, the IR reverb). Unknown ids fail loudly: silent success is
        // how agents ship projects that look right and render wrong.
        const auto effect = resolveInternalEffect(pluginId);
        if (!effect)
            return fail("Unknown internal effect \"" + pluginId +
                        "\". Known effects: " + knownEffectHint());
        auto name = pluginId;
        bool named = false;
        bool preFader = false;
        while (index < static_cast<size_t>(tokens.size())) {
            const auto token = tokens[static_cast<int>(index)];
            if (token == "--pre" || token == "--post") {
                preFader = token == "--pre";
                ++index;
            } else if (!named && !isCommand(token)) {
                name = token;
                named = true;
                ++index;
            } else {
                break;
            }
        }

        juce::String error;
        const auto deviceId =
            addInternalEffect(*trackId, *effect, name, preFader || effect->sidechainDevice, error);
        if (deviceId == magda::INVALID_DEVICE_ID)
            return fail("Failed to add internal effect: " + error);
        std::cout << "fx " << deviceId << "\n";
        return {};
    }

    CommandResult setDeviceParameter(const juce::StringArray& tokens, size_t& index) {
        auto deviceId = takeInt(tokens, index,
                                "set-device-parameter requires <device-id>");
        if (!deviceId)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("set-device-parameter requires <param-index|param-name>");

        // Accept either a numeric parameter index or a parameter name/stableId.
        const auto paramToken = tokens[static_cast<int>(index++)];
        const auto parsedIndex = parseInt(paramToken);
        std::optional<int> paramIndex;
        juce::String paramName;
        bool byName = false;
        if (parsedIndex.has_value()) {
            paramIndex = parsedIndex;
        } else {
            byName = true;
            paramName = paramToken.trim();
            if (paramName.isEmpty())
            return fail(lastParseError_);
        }
        auto value = takeDouble(tokens, index, "set-device-parameter requires <value>");
        if (!value)
            return fail(lastParseError_);

        auto& trackManager = magda::TrackManager::getInstance();
        const auto path = trackManager.findDevicePath(*deviceId);
        if (!path.isValid())
            return fail("set-device-parameter: unknown device " + juce::String(*deviceId));

        auto* device = trackManager.getDeviceInChainByPath(path);
        if (device == nullptr)
            return fail("set-device-parameter: device " + juce::String(*deviceId) +
                        " not found at its resolved path");

        int requestedIndex = -1;
        if (byName) {
            // The plugin's own parameters win; the engine-wrapper Dry/Wet pair
            // is reachable as "host:<name>" (or by its name when the plugin has
            // no parameter of that name).
            const bool hostOnly = paramName.startsWithIgnoreCase("host:");
            const auto wanted = hostOnly ? paramName.substring(5).trim() : paramName;
            if (!hostOnly) {
                for (const auto& p : device->parameters) {
                    if (p.stableId.equalsIgnoreCase(wanted) || p.name.equalsIgnoreCase(wanted)) {
                        requestedIndex = p.paramIndex;
                        break;
                    }
                }
            }
            if (requestedIndex < 0) {
                for (const auto& p : device->wrapperParameters) {
                    if (p.name.equalsIgnoreCase(wanted)) {
                        requestedIndex = p.paramIndex;
                        break;
                    }
                }
            }
            if (requestedIndex < 0) {
                juce::StringArray names;
                for (const auto& p : device->parameters)
                    names.add(p.name.isNotEmpty() ? p.name : p.stableId);
                for (const auto& p : device->wrapperParameters)
                    names.add("host:" + p.name);
                return fail("set-device-parameter: device " + juce::String(*deviceId) +
                            " has no parameter named \"" + paramName +
                            "\" (available: " +
                            (names.isEmpty() ? juce::String("<none populated yet>")
                                             : names.joinIntoString(", ")) +
                            ")");
            }
        } else {
            requestedIndex = *paramIndex;
        }
        const auto* param = device->findParameterByIndex(requestedIndex);
        if (param == nullptr) {
            juce::StringArray available;
            for (const auto& p : device->parameters)
                available.add(juce::String(p.paramIndex));
            return fail("set-device-parameter: device " + juce::String(*deviceId) +
                        " has no parameter index " + juce::String(requestedIndex) +
                        " (available: " +
                        (available.isEmpty() ? juce::String("<none populated yet>")
                                             : available.joinIntoString(", ")) +
                        ")");
        }

        const float clamped = juce::jlimit(param->minValue, param->maxValue,
                                           static_cast<float>(*value));
        trackManager.setDeviceParameterValue(path, requestedIndex,
                                            magda::ParameterModelValue{clamped});
        std::cout << "param device=" << *deviceId << " index=" << requestedIndex
                  << " value=" << clamped << "\n";
        return {};
    }

    CommandResult addMidiClip(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeInt(tokens, index, "add-midi-clip requires <track-id>");
        if (!trackId)
            return fail(lastParseError_);
        auto start = takeDouble(tokens, index, "add-midi-clip requires <start-beats>");
        if (!start)
            return fail(lastParseError_);
        auto length = takeDouble(tokens, index, "add-midi-clip requires <length-beats>");
        if (!length)
            return fail(lastParseError_);
        if (*length <= 0.0)
            return fail("add-midi-clip requires a positive length");
        if (*start < 0.0)
            return fail("add-midi-clip requires a start >= 0");
        if (magda::TrackManager::getInstance().getTrack(*trackId) == nullptr)
            return fail("add-midi-clip: unknown track " + juce::String(*trackId));

        // Clips on one track never stack. The engine's default pushes a new
        // clip past whatever it would overlap; --strict refuses instead and
        // --replace trims/removes what is underneath.
        auto policy = magda::ClipOverlapPolicy::PreserveExisting;
        bool strict = false;
        while (index < static_cast<size_t>(tokens.size())) {
            const auto token = tokens[static_cast<int>(index)];
            if (token == "--replace") {
                policy = magda::ClipOverlapPolicy::ResolveOverlaps;
                ++index;
            } else if (token == "--strict") {
                strict = true;
                ++index;
            } else {
                break;
            }
        }
        if (strict) {
            const auto hits = overlappingClips(*trackId, *start, *length, magda::INVALID_CLIP_ID);
            if (!hits.empty())
                return fail("add-midi-clip: the clip would overlap " + describeClipSpans(hits));
        }

        auto& clipManager = magda::ClipManager::getInstance();
        auto id = clipManager.createMidiClipBeats(*trackId, *start, *length,
                                                  magda::ClipView::Arrangement, policy);
        std::cout << "clip " << id << "\n";
        if (const auto* clip = clipManager.getClip(id);
            clip != nullptr && std::abs(clip->placement.startBeat - *start) > 1.0e-9)
            std::cout << "shifted " << *start << " -> " << clip->placement.startBeat << "\n";
        return {};
    }

    CommandResult moveClipCommand(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "move-clip requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        auto start = takeDouble(tokens, index, "move-clip requires <start-beats>");
        if (!start)
            return fail(lastParseError_);
        std::optional<magda::TrackId> track;
        bool replace = false;
        if (!takeClipFlags(tokens, index, &track, replace))
            return fail(lastParseError_);
        juce::String error;
        if (!moveClipChecked(*clipId, track, *start, replace, error))
            return fail("move-clip: " + error);
        std::cout << "moved " << *clipId << "\n";
        return {};
    }

    CommandResult resizeClipCommand(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "resize-clip requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        auto length = takeDouble(tokens, index, "resize-clip requires <length-beats>");
        if (!length)
            return fail(lastParseError_);
        bool replace = false;
        if (!takeClipFlags(tokens, index, nullptr, replace))
            return fail(lastParseError_);
        juce::String error;
        if (!resizeClipChecked(*clipId, *length, replace, error))
            return fail("resize-clip: " + error);
        std::cout << "resized " << *clipId << "\n";
        return {};
    }

    CommandResult duplicateClipCommand(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "duplicate-clip requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        auto start = takeDouble(tokens, index, "duplicate-clip requires <start-beats>");
        if (!start)
            return fail(lastParseError_);
        std::optional<magda::TrackId> track;
        bool replace = false;
        if (!takeClipFlags(tokens, index, &track, replace))
            return fail(lastParseError_);
        juce::String error;
        const auto copy =
            duplicateClipChecked(*clipId, track, *start, replace ? "replace" : "error", error);
        if (copy == magda::INVALID_CLIP_ID)
            return fail("duplicate-clip: " + error);
        std::cout << "clip " << copy << "\n";
        return {};
    }

    CommandResult renameClipCommand(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "rename-clip requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        const auto name = takeWords(tokens, index);
        if (name.isEmpty())
            return fail("rename-clip requires <name>");
        if (magda::ClipManager::getInstance().getClip(*clipId) == nullptr)
            return fail("rename-clip: unknown clip " + juce::String(*clipId));
        magda::ClipManager::getInstance().setClipName(*clipId, name);
        std::cout << "renamed " << *clipId << "\n";
        return {};
    }

    CommandResult setClipLoopCommand(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "set-clip-loop requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        auto enabled = takeSwitch(tokens, index, "set-clip-loop requires <on|off>");
        if (!enabled)
            return fail(lastParseError_);
        std::optional<double> loopLength;
        std::optional<double> totalLength;
        bool replace = false;
        while (index < static_cast<size_t>(tokens.size())) {
            const auto token = tokens[static_cast<int>(index)];
            if (token == "--length") {
                ++index;
                totalLength = takeDouble(tokens, index, "--length requires <total-beats>");
                if (!totalLength)
                    return fail(lastParseError_);
            } else if (token == "--replace") {
                replace = true;
                ++index;
            } else if (!loopLength && !isCommand(token)) {
                loopLength = parseDouble(token);
                if (!loopLength)
                    return fail("set-clip-loop loop length must be a number of beats");
                ++index;
            } else {
                break;
            }
        }
        juce::String error;
        if (!setClipLoopChecked(*clipId, *enabled, loopLength, totalLength, replace, error))
            return fail("set-clip-loop: " + error);
        std::cout << "loop " << *clipId << " " << (*enabled ? "on" : "off") << "\n";
        return {};
    }

    // Automation targets on the command line: volume:<track>, pan:<track>,
    // send:<track>:<dest-track>, param:<device-id>:<param-index-or-name>.
    static juce::var parseAutomationTargetToken(const juce::String& token) {
        const auto kind = token.upToFirstOccurrenceOf(":", false, false).trim().toLowerCase();
        const auto rest = token.fromFirstOccurrenceOf(":", false, false);
        if (rest.isEmpty())
            return {};
        auto* obj = new juce::DynamicObject();
        juce::var target(obj);
        obj->setProperty("kind", kind);
        if (kind == "volume" || kind == "pan") {
            obj->setProperty("trackId", rest.trim());
        } else if (kind == "send") {
            obj->setProperty("trackId", rest.upToFirstOccurrenceOf(":", false, false).trim());
            obj->setProperty("to", rest.fromFirstOccurrenceOf(":", false, false).trim());
        } else if (kind == "param") {
            const auto deviceId = parseInt(rest.upToFirstOccurrenceOf(":", false, false));
            const auto param = rest.fromFirstOccurrenceOf(":", false, false);
            if (!deviceId || param.isEmpty())
                return {};
            obj->setProperty("deviceId", *deviceId);
            obj->setProperty("param", param);
        } else {
            return {};
        }
        return target;
    }

    CommandResult addAutomationPoint(const juce::StringArray& tokens, size_t& index) {
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("add-automation-point requires <target>");
        const auto target = parseAutomationTargetToken(tokens[static_cast<int>(index++)]);
        if (target.isVoid())
            return fail("automation targets look like volume:<track>, pan:<track>, "
                        "send:<track>:<dest-track> or param:<device-id>:<param>");
        auto beat = takeDouble(tokens, index, "add-automation-point requires <beat>");
        if (!beat)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("add-automation-point requires <value>");
        const auto valueToken = tokens[static_cast<int>(index++)];
        juce::var value;
        if (const auto number = parseDouble(valueToken))
            value = *number;
        else
            value = valueToken;
        auto* point = new juce::DynamicObject();
        juce::var pointVar(point);
        point->setProperty("beat", *beat);
        point->setProperty("value", value);
        if (index < static_cast<size_t>(tokens.size()) &&
            parseCurve(tokens[static_cast<int>(index)]))
            point->setProperty("curve", tokens[static_cast<int>(index++)]);
        juce::Array<juce::var> points;
        points.add(pointVar);
        auto* request = new juce::DynamicObject();
        juce::var requestVar(request);
        request->setProperty("target", target);
        request->setProperty("points", points);
        request->setProperty("mode", "add");
        juce::DynamicObject::Ptr result = new juce::DynamicObject();
        const auto error = applyAutomationRequest(requestVar, *result);
        if (error.isNotEmpty())
            return fail("add-automation-point: " + error);
        std::cout << "automation lane " << result->getProperty("laneId").toString() << " points "
                  << result->getProperty("points").toString() << "\n";
        return {};
    }

    CommandResult clearAutomation(const juce::StringArray& tokens, size_t& index) {
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("clear-automation requires <target>");
        const auto target = parseAutomationTargetToken(tokens[static_cast<int>(index++)]);
        if (target.isVoid())
            return fail("automation targets look like volume:<track>, pan:<track>, "
                        "send:<track>:<dest-track> or param:<device-id>:<param>");
        auto* request = new juce::DynamicObject();
        juce::var requestVar(request);
        request->setProperty("target", target);
        request->setProperty("mode", "clear");
        juce::DynamicObject::Ptr result = new juce::DynamicObject();
        const auto error = applyAutomationRequest(requestVar, *result);
        if (error.isNotEmpty())
            return fail("clear-automation: " + error);
        std::cout << "automation cleared\n";
        return {};
    }

    CommandResult deleteClip(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "delete-clip requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        engine_.getMagdaApi().clips().deleteClip(*clipId);
        return {};
    }

    struct ParsedNote {
        int clipId = 0;
        magda::MidiNote note;
    };

    std::optional<ParsedNote> parseNoteArgs(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "add-midi-note requires <clip-id>");
        if (!clipId)
            return std::nullopt;
        auto start = takeDouble(tokens, index, "add-midi-note requires <start-beats>");
        if (!start)
            return std::nullopt;
        auto noteNumber = takeInt(tokens, index, "add-midi-note requires <note>");
        if (!noteNumber)
            return std::nullopt;
        auto length = takeDouble(tokens, index, "add-midi-note requires <length-beats>");
        if (!length)
            return std::nullopt;

        int velocity = 100;
        if (index < static_cast<size_t>(tokens.size()) &&
            !isCommand(tokens[static_cast<int>(index)])) {
            auto parsedVelocity = parseInt(tokens[static_cast<int>(index)]);
            if (!parsedVelocity) {
                lastParseError_ = "add-midi-note velocity must be an integer";
                return std::nullopt;
            }
            velocity = juce::jlimit(0, 127, *parsedVelocity);
            ++index;
        }

        ParsedNote parsed;
        parsed.clipId = *clipId;
        parsed.note.startBeat = *start;
        parsed.note.noteNumber = *noteNumber;
        parsed.note.lengthBeats = *length;
        parsed.note.velocity = velocity;
        return parsed;
    }

    CommandResult addMidiNote(const juce::StringArray& tokens, size_t& index) {
        auto first = parseNoteArgs(tokens, index);
        if (!first)
            return fail(lastParseError_);

        // Consecutive add-midi-note commands for the same clip become ONE
        // undoable command with ONE clip notification. One command per note
        // re-syncs the clip after every note, which is quadratic for long parts.
        std::vector<magda::MidiNote> batch{first->note};
        while (index < static_cast<size_t>(tokens.size()) &&
               tokens[static_cast<int>(index)] == "add-midi-note") {
            size_t cursor = index + 1;
            if (cursor >= static_cast<size_t>(tokens.size()))
                break;
            const auto nextClip = parseInt(tokens[static_cast<int>(cursor)]);
            if (!nextClip || *nextClip != first->clipId)
                break;
            auto parsed = parseNoteArgs(tokens, cursor);
            if (!parsed)
                return fail(lastParseError_);
            batch.push_back(parsed->note);
            index = cursor;
        }

        if (batch.size() == 1) {
            const auto& note = batch.front();
            if (!engine_.getMagdaApi().clips().addMidiNote(first->clipId, note.startBeat,
                                                           note.noteNumber, note.lengthBeats,
                                                           note.velocity))
                return fail("Failed to add MIDI note");
            std::cout << "note\n";
            return {};
        }

        const auto* clip = magda::ClipManager::getInstance().getClip(first->clipId);
        if (clip == nullptr || !clip->isMidi())
            return fail("Failed to add MIDI note");
        for (const auto& note : batch)
            if (note.lengthBeats <= 0.0)
                return fail("Failed to add MIDI note");

        const auto count = batch.size();
        {
            magda::ClipManager::BatchScope notifications;
            magda::UndoManager::getInstance().executeCommand(
                std::make_unique<magda::AddMultipleMidiNotesCommand>(first->clipId,
                                                                     std::move(batch),
                                                                     "Add MIDI Notes"));
        }
        std::cout << "notes " << count << "\n";
        return {};
    }

    CommandResult quantizeNotes(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "quantize-notes requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        auto grid = takeDouble(tokens, index, "quantize-notes requires <grid-beats>");
        if (!grid)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("quantize-notes requires <start|length|both>");

        auto mode = parseQuantizeMode(tokens[static_cast<int>(index++)]);
        if (!mode)
            return fail("quantize-notes mode must be start, length, or both");

        auto indices = takeNoteIndices(tokens, index, *clipId, "quantize-notes");
        if (indices.empty())
            return fail("quantize-notes requires at least one note index or all");

        if (!engine_.getMagdaApi().clips().quantizeMidiNotes(*clipId, indices, *grid, *mode))
            return fail("Failed to quantize MIDI notes");
        return {};
    }

    CommandResult sliceNotes(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "slice-notes requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        auto subdivisions = takeInt(tokens, index, "slice-notes requires <subdivisions>");
        if (!subdivisions)
            return fail(lastParseError_);

        auto indices = takeNoteIndices(tokens, index, *clipId, "slice-notes");
        if (indices.empty())
            return fail("slice-notes requires at least one note index or all");

        if (!engine_.getMagdaApi().clips().sliceMidiNotes(*clipId, indices, *subdivisions))
            return fail("Failed to slice MIDI notes");
        return {};
    }

    CommandResult transposeMidiClip(const juce::StringArray& tokens, size_t& index) {
        auto clipId = takeInt(tokens, index, "transpose-midi-clip requires <clip-id>");
        if (!clipId)
            return fail(lastParseError_);
        auto semitones = takeInt(tokens, index, "transpose-midi-clip requires <semitones>");
        if (!semitones)
            return fail(lastParseError_);

        if (!engine_.getMagdaApi().clips().transposeMidiClip(*clipId, *semitones))
            return fail("Failed to transpose MIDI clip");
        return {};
    }

    CommandResult pluginsList(const juce::StringArray& tokens, size_t& index) {
        juce::ignoreUnused(tokens, index);
        for (const auto& desc : engine_.getKnownPluginTypes()) {
            std::cout << "plugin " << desc.name << "\t" << desc.pluginFormatName << "\t"
                      << (desc.isInstrument ? "instrument" : "effect") << "\t"
                      << desc.createIdentifierString() << "\n";
        }
        return {};
    }

    // Blocking out-of-process plugin scan (first-run bootstrap). The engine
    // normally scans lazily in the GUI app; headless runs must ask explicitly.
    CommandResult scanPlugins(const juce::StringArray& tokens, size_t& index) {
        juce::ignoreUnused(tokens, index);
        auto* wrapper = asTracktionWrapper(engine_);
        if (wrapper == nullptr)
            return fail("plugin scanning requires the Tracktion engine");

        std::promise<bool> done;
        auto finished = done.get_future();
        wrapper->setPluginScanCompletionCallback(
            [&done](bool success, int, const juce::StringArray&) {
                done.set_value(success);
            });
        wrapper->startPluginScan();

        // Pump the message loop while the scan workers report back.
        if (auto* mm = juce::MessageManager::getInstance()) {
            while (finished.wait_for(std::chrono::milliseconds(50)) !=
                   std::future_status::ready) {
                mm->runDispatchLoopUntil(50);
                if (!wrapper->isPluginScanRunning()
                    && finished.wait_for(std::chrono::milliseconds(0))
                           == std::future_status::ready)
                    break;
            }
        } else {
            finished.wait();
        }
        wrapper->setPluginScanCompletionCallback(nullptr);

        const auto known = engine_.getKnownPluginTypes();
        std::cout << "scanned " << known.size() << " plugins\n";
        for (const auto& desc : known) {
            std::cout << "plugin " << desc.name << "\t" << desc.pluginFormatName
                      << "\t" << (desc.isInstrument ? "instrument" : "effect")
                      << "\t" << desc.createIdentifierString() << "\n";
        }
        return {};
    }

    CommandResult addPlugin(const juce::StringArray& tokens, size_t& index) {
        auto trackId = takeInt(tokens, index, "add-plugin requires <track-id>");
        if (!trackId)
            return fail(lastParseError_);
        if (index >= static_cast<size_t>(tokens.size()))
            return fail("add-plugin requires <plugin-id-or-name>");

        // The plugin id/name is one token in the common case; tolerate names
        // split across tokens by joining plain words until a flag/command.
        juce::StringArray queryParts;
        bool asInstrument = false;
        while (index < static_cast<size_t>(tokens.size())) {
            const auto token = tokens[static_cast<int>(index)];
            if (token == "--instrument") {
                asInstrument = true;
                ++index;
            } else if (token == "--out") {
                ++index;  // generic exec loop owns --out; don't choke on it
                if (index < static_cast<size_t>(tokens.size()))
                    ++index;
            } else if (isCommand(token)) {
                break;
            } else {
                queryParts.add(token);
                ++index;
            }
        }
        if (queryParts.isEmpty())
            return fail("add-plugin requires <plugin-id-or-name>");

        auto description = findExternalPluginDescription(engine_, queryParts.joinIntoString(" "));
        if (!description && queryParts.size() > 1)
            description = findExternalPluginDescription(engine_, queryParts[0]);
        if (!description)
            return fail("unknown plugin: " + queryParts.joinIntoString(" "));

        auto* wrapper = asTracktionWrapper(engine_);
        if (wrapper == nullptr)
            return fail("external plugins require the Tracktion engine");

        const auto extId = wrapper->addExternalPlugin(std::to_string(*trackId), *description,
                                                      asInstrument);
        if (extId <= 0)
            return fail("failed to instantiate plugin");
        std::cout << "device " << extId << "\n";
        return {};
    }

    CommandResult removeDevice(const juce::StringArray& tokens, size_t& index) {
        auto extId = takeInt(tokens, index, "remove-device requires <ext-id>");
        if (!extId)
            return fail(lastParseError_);
        auto* wrapper = asTracktionWrapper(engine_);
        if (wrapper == nullptr || !wrapper->removeExternalPlugin(*extId))
            return fail("unknown device");
        std::cout << "removed " << *extId << "\n";
        return {};
    }

    CommandResult extParameters(const juce::StringArray& tokens, size_t& index) {
        auto extId = takeInt(tokens, index, "ext-parameters requires <ext-id>");
        if (!extId)
            return fail(lastParseError_);
        auto* wrapper = asTracktionWrapper(engine_);
        if (wrapper == nullptr)
            return fail("external plugins require the Tracktion engine");
        const auto* entry = findExternalPluginEntry(*wrapper, *extId);
        if (entry == nullptr)
            return fail("unknown device");
        for (const auto& info : wrapper->getExternalPluginParameters(*extId))
            std::cout << "param " << info.name << "\t" << info.normalised << "\n";
        return {};
    }

    CommandResult setExtParameter(const juce::StringArray& tokens, size_t& index) {
        auto extId = takeInt(tokens, index, "set-ext-parameter requires <ext-id>");
        if (!extId)
            return fail(lastParseError_);

        // <param-name> may contain spaces: join plain tokens until a flag;
        // the final joined token must be the numeric value.
        juce::StringArray parts;
        while (index < static_cast<size_t>(tokens.size()) &&
               !isCommand(tokens[static_cast<int>(index)]))
            parts.add(tokens[static_cast<int>(index++)]);
        if (parts.size() < 2)
            return fail("set-ext-parameter requires <param-name> <normalised 0..1>");

        const auto value = parseDouble(parts[parts.size() - 1]);
        if (!value)
            return fail("set-ext-parameter value must be a number in [0.0, 1.0]");
        if (*value < 0.0 || *value > 1.0)
            return fail("set-ext-parameter value must be a number in [0.0, 1.0]");

        juce::StringArray nameParts;
        for (int i = 0; i < parts.size() - 1; ++i)
            nameParts.add(parts[i]);

        auto* wrapper = asTracktionWrapper(engine_);
        if (wrapper == nullptr ||
            !wrapper->setExternalPluginParameter(*extId, nameParts.joinIntoString(" "),
                                                 static_cast<float>(*value)))
            return fail("unknown device or parameter");
        std::cout << "param set\n";
        return {};
    }

    static const magda::ExternalPluginEntry* findExternalPluginEntry(
        magda::TracktionEngineWrapper& wrapper, int extId) {
        const auto* entries = wrapper.externalPluginList();
        if (entries == nullptr)
            return nullptr;
        for (const auto& entry : *entries)
            if (entry.extId == extId)
                return &entry;
        return nullptr;
    }

    CommandResult dump(const juce::StringArray& tokens, size_t& index) {
        if (index >= static_cast<size_t>(tokens.size()) ||
            tokens[static_cast<int>(index)] != "--json")
            return fail("dump requires --json");
        ++index;
        dumpJson();
        return {};
    }

    static bool isCommand(const juce::String& token) {
        for (const auto& spec : commandSpecs())
            if (token == spec.name)
                return true;
        return token.startsWith("--");
    }

    static std::optional<magda::MidiNoteQuantizeMode> parseQuantizeMode(const juce::String& token) {
        auto normalized = token.trim().toLowerCase();
        if (normalized == "start")
            return magda::MidiNoteQuantizeMode::StartOnly;
        if (normalized == "length")
            return magda::MidiNoteQuantizeMode::LengthOnly;
        if (normalized == "both" || normalized == "start-and-length")
            return magda::MidiNoteQuantizeMode::StartAndLength;
        return std::nullopt;
    }

    std::vector<size_t> takeNoteIndices(const juce::StringArray& tokens, size_t& index,
                                        magda::ClipId clipId, const juce::String& commandName) {
        std::vector<size_t> indices;
        if (index >= static_cast<size_t>(tokens.size()) ||
            isCommand(tokens[static_cast<int>(index)])) {
            lastParseError_ = commandName + " requires <all|note-index...>";
            return indices;
        }

        if (tokens[static_cast<int>(index)] == "all") {
            ++index;
            if (auto* clip = engine_.getMagdaApi().clips().getClip(clipId)) {
                indices.reserve(clip->midiNotes.size());
                for (size_t i = 0; i < clip->midiNotes.size(); ++i)
                    indices.push_back(i);
            }
            return indices;
        }

        while (index < static_cast<size_t>(tokens.size()) &&
               !isCommand(tokens[static_cast<int>(index)])) {
            auto parsed = parseInt(tokens[static_cast<int>(index)]);
            if (!parsed || *parsed < 0) {
                lastParseError_ = commandName + " note indices must be non-negative integers";
                indices.clear();
                return indices;
            }
            indices.push_back(static_cast<size_t>(*parsed));
            ++index;
        }
        return indices;
    }

    std::optional<int> takeInt(const juce::StringArray& tokens, size_t& index,
                               const juce::String& missingError) {
        if (index >= static_cast<size_t>(tokens.size())) {
            lastParseError_ = missingError;
            return std::nullopt;
        }
        auto value = parseInt(tokens[static_cast<int>(index++)]);
        if (!value)
            lastParseError_ = "Expected integer argument";
        return value;
    }

    std::optional<double> takeDouble(const juce::StringArray& tokens, size_t& index,
                                     const juce::String& missingError) {
        if (index >= static_cast<size_t>(tokens.size())) {
            lastParseError_ = missingError;
            return std::nullopt;
        }
        auto value = parseDouble(tokens[static_cast<int>(index++)]);
        if (!value)
            lastParseError_ = "Expected numeric argument";
        return value;
    }

    // A track id, or "master" when allowed; the track must exist.
    std::optional<magda::TrackId> takeTrack(const juce::StringArray& tokens, size_t& index,
                                            const juce::String& missingError, bool allowMaster) {
        if (index >= static_cast<size_t>(tokens.size())) {
            lastParseError_ = missingError;
            return std::nullopt;
        }
        const auto token = tokens[static_cast<int>(index++)];
        const auto trackId = parseTrackRef(token);
        if (!trackId) {
            lastParseError_ = missingError + " (got \"" + token + "\")";
            return std::nullopt;
        }
        if (*trackId == magda::MASTER_TRACK_ID) {
            if (!allowMaster) {
                lastParseError_ = missingError + " (the master is not allowed here)";
                return std::nullopt;
            }
            return trackId;
        }
        if (magda::TrackManager::getInstance().getTrack(*trackId) == nullptr) {
            lastParseError_ = "unknown track " + juce::String(*trackId);
            return std::nullopt;
        }
        return trackId;
    }

    std::optional<bool> takeSwitch(const juce::StringArray& tokens, size_t& index,
                                   const juce::String& missingError) {
        if (index >= static_cast<size_t>(tokens.size())) {
            lastParseError_ = missingError;
            return std::nullopt;
        }
        const auto value = parseSwitch(tokens[static_cast<int>(index++)]);
        if (!value)
            lastParseError_ = missingError;
        return value;
    }

    // Plain words up to the next command or flag, joined by spaces.
    juce::String takeWords(const juce::StringArray& tokens, size_t& index) {
        juce::StringArray words;
        while (index < static_cast<size_t>(tokens.size()) &&
               !isCommand(tokens[static_cast<int>(index)]))
            words.add(tokens[static_cast<int>(index++)]);
        return words.joinIntoString(" ").trim();
    }

    // [--track <id>] [--replace] after a clip command.
    bool takeClipFlags(const juce::StringArray& tokens, size_t& index,
                       std::optional<magda::TrackId>* track, bool& replace) {
        while (index < static_cast<size_t>(tokens.size())) {
            const auto token = tokens[static_cast<int>(index)];
            if (token == "--replace") {
                replace = true;
                ++index;
            } else if (token == "--track" && track != nullptr) {
                ++index;
                *track = takeTrack(tokens, index, "--track requires <track-id>", false);
                if (!*track)
                    return false;
            } else {
                break;
            }
        }
        return true;
    }

    static CommandResult fail(const juce::String& error) {
        return {false, error};
    }

    magda::AudioEngine& engine_;
    juce::String lastParseError_;
};

void printUsage(std::ostream& out) {
    out << "magda-cli " << MAGDA_VERSION << "\n"
        << "\n"
        << "Usage:\n"
        << "  magda-cli boot\n"
        << "  magda-cli init <out.mgd>\n"
        << "  magda-cli run <project.mgd> [--out <out.mgd>]\n"
        << "  magda-cli run <project.mgd> --cmds <cmds.txt> [--out <out.mgd>] [--dump-json]\n"
        << "  magda-cli exec <project.mgd> <commands...> [--out <out.mgd>] [--dump-json]\n"
        << "  magda-cli render <project.mgd> --wav <out.wav> [--from <time>] [--to <time>]\n"
        << "        [--tracks <id,id,...> | --stem <id>] [--no-master-fx|--master-fx]\n"
        << "  magda-cli serve <project.mgd> [--sock <path>]\n"
        << "  magda-cli commands [--json]\n"
        << "\n"
        << "Commands:\n";

    for (const auto& spec : CommandDispatcher::commandSpecs())
        if (spec.showInUsage)
            out << "  " << spec.usage << "\n";
}

struct RunOptions {
    juce::File input;
    juce::File output;
    juce::File commandFile;
    bool hasCommandFile = false;
    bool dumpJson = false;
    juce::StringArray execTokens;
};

struct RenderOptions {
    juce::File input;
    juce::File wavOutput;
    std::optional<double> fromSeconds;
    std::optional<double> toSeconds;
    std::optional<double> sampleRate;
    std::optional<int> bitDepth;
    std::vector<magda::TrackId> trackIds;     // render only these tracks (engine track filter)
    std::optional<magda::TrackId> stemTrack;  // one channel on its own (see StemRenderScope)
    std::optional<bool> useMasterPlugins;     // master fader + master inserts; default: not for stems
};

bool loadProjectForCli(const juce::File& input, HeadlessEngineSession& session) {
    auto& projectManager = magda::ProjectManager::getInstance();
    if (!projectManager.loadProject(input, [&session](const magda::ProjectInfo& info) {
            restoreProjectTiming(session.engine(), info);
        })) {
        std::cerr << "Failed to load project: " << projectManager.getLastError() << "\n";
        return false;
    }
    return true;
}

std::optional<double> parseRenderTime(const juce::String& text, const magda::ProjectInfo& info) {
    auto token = text.trim().toLowerCase();
    if (token.endsWith("bars")) {
        auto bars = parseDouble(token.dropLastCharacters(4));
        if (!bars)
            return std::nullopt;
        const double beats = *bars * info.timeSignatureNumerator;
        return beats * 60.0 / info.tempo;
    }
    if (token.endsWith("bar")) {
        auto bars = parseDouble(token.dropLastCharacters(3));
        if (!bars)
            return std::nullopt;
        const double beats = *bars * info.timeSignatureNumerator;
        return beats * 60.0 / info.tempo;
    }
    if (token.endsWith("beats")) {
        auto beats = parseDouble(token.dropLastCharacters(5));
        if (!beats)
            return std::nullopt;
        return *beats * 60.0 / info.tempo;
    }
    if (token.endsWith("beat")) {
        auto beats = parseDouble(token.dropLastCharacters(4));
        if (!beats)
            return std::nullopt;
        return *beats * 60.0 / info.tempo;
    }
    if (token.endsWith("s"))
        token = token.dropLastCharacters(1);
    return parseDouble(token);
}

double defaultRenderEndSeconds(magda::AudioEngine& engine, const magda::ProjectInfo& info) {
    const auto editLength = engine.getEditLengthBeats();
    if (editLength.value > 0.0)
        if (const auto* tempoMap = engine.tempoMap())
            return tempoMap->beatToTime(editLength.value);

    const double beats = static_cast<double>(info.timelineLengthBars) * info.timeSignatureNumerator;
    const double seconds = beats * 60.0 / info.tempo;
    return juce::jmax(1.0, seconds);
}

bool prepareOutputFile(const juce::File& file) {
    if (!file.getParentDirectory().createDirectory()) {
        std::cerr << "Failed to create output directory: "
                  << file.getParentDirectory().getFullPathName() << "\n";
        return false;
    }

    if (file.exists() && !file.deleteFile()) {
        std::cerr << "Failed to replace output file: " << file.getFullPathName() << "\n";
        return false;
    }

    return true;
}

bool anyDeviceLoading();

// Lets timer-driven engine work land before an offline render: automation
// lanes are baked into the engine's curves on the audio bridge's 30 Hz timer,
// and hosted plugins restored with a project finish loading asynchronously.
void settleForRender(int minimumMs) {
    auto* messageManager = juce::MessageManager::getInstance();
    if (messageManager == nullptr)
        return;
    const auto settle = juce::Time::getMillisecondCounter() + static_cast<juce::uint32>(minimumMs);
    const auto deadline = juce::Time::getMillisecondCounter() + 60000;
    while (juce::Time::getMillisecondCounter() < settle ||
           (anyDeviceLoading() && juce::Time::getMillisecondCounter() < deadline))
        messageManager->runDispatchLoopUntil(20);
}

// Reshapes the mixer for the length of one stem render so the offline render
// captures that channel on its own: solos are cleared, a stem that feeds
// another track (a group child) is routed straight to the master, and the
// tracks it depends on without being part of it -- the senders into an aux
// stem, the sidechain sources keying its devices -- join the render muted.
// (Muted tracks still process in this engine, so their sends and keys stay
// live.) Everything is put back on destruction; nothing is saved meanwhile.
class StemRenderScope {
  public:
    explicit StemRenderScope(magda::TrackId stem) {
        auto& trackManager = magda::TrackManager::getInstance();
        const auto* stemTrack = trackManager.getTrack(stem);
        if (stem == magda::MASTER_TRACK_ID || stemTrack == nullptr)
            return;
        const auto stemBus = stemTrack->auxBusIndex;
        renderSet_.push_back(stem);

        // Tracks whose audio reaches the stem by output routing (group children).
        std::set<magda::TrackId> inputs;
        std::vector<magda::TrackId> pending{stem};
        while (!pending.empty()) {
            const auto routedTo = "track:" + juce::String(pending.back());
            pending.pop_back();
            for (const auto& track : trackManager.getTracks())
                if (track.audioOutputDevice == routedTo && inputs.insert(track.id).second)
                    pending.push_back(track.id);
        }

        std::set<magda::TrackId> helpers;
        for (const auto& track : trackManager.getTracks())
            for (const auto& send : track.sends)
                if (send.destTrackId == stem || (stemBus >= 0 && send.busIndex == stemBus))
                    helpers.insert(track.id);
        std::set<magda::TrackId> keyed = inputs;
        keyed.insert(stem);
        // Sidechain sources of everything that plays, transitively: a sender
        // whose own ducker is keyed by a third track needs that track too.
        for (bool grew = true; grew;) {
            grew = false;
            std::set<magda::TrackId> sources;
            for (const auto id : keyed)
                if (const auto* track = trackManager.getTrack(id))
                    collectSidechainSources(track->chain.fxChainElements, sources);
            for (const auto id : helpers)
                if (const auto* track = trackManager.getTrack(id))
                    collectSidechainSources(track->chain.fxChainElements, sources);
            for (const auto id : sources)
                if (id != magda::INVALID_TRACK_ID && !keyed.count(id) && helpers.insert(id).second)
                    grew = true;
        }
        for (const auto id : keyed)
            helpers.erase(id);

        for (const auto& track : trackManager.getTracks())
            if (track.soloed)
                soloed_.push_back(track.id);
        for (const auto id : soloed_)
            trackManager.setTrackSoloed(id, false);

        for (const auto id : helpers) {
            const auto* track = trackManager.getTrack(id);
            if (track == nullptr || id == magda::MASTER_TRACK_ID)
                continue;
            renderSet_.push_back(id);
            if (!track->muted) {
                muted_.push_back(id);
                trackManager.setTrackMuted(id, true);
            }
            rerouteToMaster(id);
        }
        rerouteToMaster(stem);
    }

    ~StemRenderScope() {
        auto& trackManager = magda::TrackManager::getInstance();
        for (const auto& [id, routing] : rerouted_)
            trackManager.setTrackAudioOutput(id, routing);
        for (const auto id : muted_)
            trackManager.setTrackMuted(id, false);
        for (const auto id : soloed_)
            trackManager.setTrackSoloed(id, true);
    }

    StemRenderScope(const StemRenderScope&) = delete;
    StemRenderScope& operator=(const StemRenderScope&) = delete;

    const std::vector<magda::TrackId>& renderSet() const {
        return renderSet_;
    }

  private:
    void rerouteToMaster(magda::TrackId id) {
        auto& trackManager = magda::TrackManager::getInstance();
        const auto* track = trackManager.getTrack(id);
        if (track == nullptr || !track->audioOutputDevice.startsWith("track:"))
            return;
        rerouted_.emplace_back(id, track->audioOutputDevice);
        trackManager.setTrackAudioOutput(id, "master");
    }

    static void collectSidechainSources(const std::vector<magda::ChainElement>& elements,
                                        std::set<magda::TrackId>& out) {
        for (const auto& element : elements) {
            if (magda::isDevice(element)) {
                const auto& device = magda::getDevice(element);
                if (device.sidechain.isActive())
                    out.insert(device.sidechain.sourceTrackId);
            } else if (magda::isRack(element)) {
                const auto& rack = magda::getRack(element);
                if (rack.sidechain.isActive())
                    out.insert(rack.sidechain.sourceTrackId);
                for (const auto& chain : rack.chains)
                    collectSidechainSources(chain.elements, out);
            }
        }
    }

    std::vector<magda::TrackId> renderSet_;
    std::vector<magda::TrackId> muted_;
    std::vector<magda::TrackId> soloed_;
    std::vector<std::pair<magda::TrackId, juce::String>> rerouted_;
};

bool renderWav(magda::AudioEngine& engine, const RenderOptions& options) {
    if (!engine.hasActiveEdit()) {
        std::cerr << "No edit is loaded for rendering\n";
        return false;
    }

    if (!prepareOutputFile(options.wavOutput))
        return false;

    // Declared before the render session so the mixer is restored only after
    // the session has been torn down.
    std::unique_ptr<StemRenderScope> stemScope;
    auto trackIds = options.trackIds;
    if (options.stemTrack) {
        stemScope = std::make_unique<StemRenderScope>(*options.stemTrack);
        trackIds = stemScope->renderSet();
        if (trackIds.empty()) {
            std::cerr << "Unknown stem track " << *options.stemTrack << "\n";
            return false;
        }
    }
    settleForRender(150);

    const auto& projectInfo = magda::ProjectManager::getInstance().getCurrentProjectInfo();
    const double startSeconds = options.fromSeconds.value_or(0.0);
    const double endSeconds =
        options.toSeconds.value_or(defaultRenderEndSeconds(engine, projectInfo));
    if (startSeconds < 0.0 || endSeconds <= startSeconds) {
        std::cerr << "Render range must have --to greater than --from\n";
        return false;
    }

    auto session = engine.createOfflineRenderSession(false);
    auto task = session ? session->createTask({
                              .destination = options.wavOutput,
                              .format = magda::OfflineRenderFormat::Wav,
                              .bitDepth = options.bitDepth.value_or(projectInfo.renderBitDepth),
                              .sampleRate = options.sampleRate.value_or(projectInfo.sampleRate),
                              .blockSize = 512,
                              .shouldNormalise = false,
                              .useMasterPlugins =
                                  options.useMasterPlugins.value_or(!options.stemTrack.has_value()),
                              .usePlugins = true,
                              .checkNodesForAudio = false,
                              .realTimeRender = false,
                              .range = {{startSeconds}, {endSeconds}, {}},
                              .trackIds = trackIds,
                              .excludedTrackIds = {},
                              .clipIds = {},
                          })
                        : nullptr;
    if (task == nullptr) {
        if (session != nullptr && !trackIds.empty())
            std::cerr << "None of the requested tracks can be rendered\n";
        else
            std::cerr << "Audio engine does not support offline rendering\n";
        return false;
    }

    const auto result = task->run();
    if (!result.success) {
        std::cerr << "Render failed: " << result.error << "\n";
        return false;
    }
    if (!options.wavOutput.existsAsFile() || options.wavOutput.getSize() <= 0) {
        std::cerr << "Render did not produce a WAV file: " << options.wavOutput.getFullPathName()
                  << "\n";
        return false;
    }

    std::cout << "Rendered " << options.wavOutput.getFullPathName() << "\n";
    return true;
}

bool saveProjectForCli(const juce::File& output) {
    auto& projectManager = magda::ProjectManager::getInstance();
    if (!projectManager.saveProjectAs(output)) {
        std::cerr << "Failed to save project: " << projectManager.getLastError() << "\n";
        return false;
    }
    std::cout << "Saved " << projectManager.getCurrentProjectFile().getFullPathName() << "\n";
    return true;
}

bool executeCommandTokens(CommandDispatcher& dispatcher, const juce::StringArray& tokens) {
    size_t index = 0;
    while (index < static_cast<size_t>(tokens.size())) {
        auto result = dispatcher.execute(tokens, index);
        if (!result.ok) {
            std::cerr << result.error << "\n";
            return false;
        }
    }
    return true;
}

bool executeCommandFile(CommandDispatcher& dispatcher, const juce::File& file) {
    if (!file.existsAsFile()) {
        std::cerr << "Command file does not exist: " << file.getFullPathName() << "\n";
        return false;
    }

    juce::StringArray lines;
    lines.addLines(file.loadFileAsString());
    for (auto line : lines) {
        line = line.trim();
        if (line.isEmpty() || line.startsWith("#"))
            continue;

        juce::StringArray tokens;
        tokens.addTokens(line, true);
        if (!executeCommandTokens(dispatcher, tokens))
            return false;
    }
    return true;
}

int runCli(const RunOptions& options) {
    HeadlessEngineSession session;
    if (!session.initialize()) {
        std::cerr << session.error() << "\n";
        return 1;
    }

    if (!loadProjectForCli(options.input, session))
        return 1;

    CommandDispatcher dispatcher(session.engine());
    if (options.hasCommandFile && !executeCommandFile(dispatcher, options.commandFile))
        return 1;
    if (!options.execTokens.isEmpty() && !executeCommandTokens(dispatcher, options.execTokens))
        return 1;
    if (options.dumpJson)
        dispatcher.dumpJson();

    if (!saveProjectForCli(options.output))
        return 1;

    return 0;
}

int initProject(const juce::StringArray& args) {
    if (args.size() != 2) {
        printUsage(std::cerr);
        return 2;
    }

    HeadlessEngineSession session;
    if (!session.initialize()) {
        std::cerr << session.error() << "\n";
        return 1;
    }

    if (!magda::ProjectManager::getInstance().newProject()) {
        std::cerr << "Failed to create project: "
                  << magda::ProjectManager::getInstance().getLastError() << "\n";
        return 1;
    }

    if (!saveProjectForCli(fileFromArg(args[1])))
        return 1;

    return 0;
}

int runRoundTrip(const juce::StringArray& args) {
    if (args.size() < 2) {
        printUsage(std::cerr);
        return 2;
    }

    RunOptions options;
    options.input = fileFromArg(args[1]);
    options.output = defaultOutputFor(options.input);

    for (int i = 2; i < args.size(); ++i) {
        if (args[i] == "--out") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            options.output = fileFromArg(args[i]);
        } else if (args[i] == "--cmds") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            options.commandFile = fileFromArg(args[i]);
            options.hasCommandFile = true;
        } else if (args[i] == "--dump-json") {
            options.dumpJson = true;
        } else {
            printUsage(std::cerr);
            return 2;
        }
    }

    return runCli(options);
}

int renderProject(const juce::StringArray& args) {
    if (args.size() < 4) {
        printUsage(std::cerr);
        return 2;
    }

    RenderOptions options;
    options.input = fileFromArg(args[1]);

    juce::String fromToken;
    juce::String toToken;
    for (int i = 2; i < args.size(); ++i) {
        if (args[i] == "--wav") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            options.wavOutput = fileFromArg(args[i]);
        } else if (args[i] == "--from") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            fromToken = args[i];
        } else if (args[i] == "--to") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            toToken = args[i];
        } else if (args[i] == "--sample-rate") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            options.sampleRate = parseDouble(args[i]);
            if (!options.sampleRate || *options.sampleRate <= 0.0) {
                std::cerr << "--sample-rate requires a positive number\n";
                return 2;
            }
        } else if (args[i] == "--bit-depth") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            options.bitDepth = parsePositiveInt(args[i]);
            if (!options.bitDepth) {
                std::cerr << "--bit-depth requires a positive integer\n";
                return 2;
            }
        } else if (args[i] == "--tracks") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            juce::StringArray ids;
            ids.addTokens(args[i], ",", "");
            for (const auto& id : ids) {
                const auto trackId = parseInt(id);
                if (!trackId || *trackId < 0) {
                    std::cerr << "--tracks takes comma-separated track ids\n";
                    return 2;
                }
                options.trackIds.push_back(*trackId);
            }
        } else if (args[i] == "--stem") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            const auto trackId = parseInt(args[i]);
            if (!trackId || *trackId < 0) {
                std::cerr << "--stem takes a track id\n";
                return 2;
            }
            options.stemTrack = *trackId;
        } else if (args[i] == "--no-master-fx" || args[i] == "--master-fx") {
            options.useMasterPlugins = args[i] == "--master-fx";
        } else {
            printUsage(std::cerr);
            return 2;
        }
    }
    if (options.stemTrack && !options.trackIds.empty()) {
        std::cerr << "render takes --tracks or --stem, not both\n";
        return 2;
    }

    if (options.wavOutput.getFullPathName().isEmpty()) {
        std::cerr << "render requires --wav <out.wav>\n";
        return 2;
    }

    HeadlessEngineSession session;
    if (!session.initialize()) {
        std::cerr << session.error() << "\n";
        return 1;
    }

    if (!loadProjectForCli(options.input, session))
        return 1;

    const auto& info = magda::ProjectManager::getInstance().getCurrentProjectInfo();
    if (fromToken.isNotEmpty()) {
        options.fromSeconds = parseRenderTime(fromToken, info);
        if (!options.fromSeconds) {
            std::cerr << "Invalid --from time: " << fromToken << "\n";
            return 2;
        }
    }
    if (toToken.isNotEmpty()) {
        options.toSeconds = parseRenderTime(toToken, info);
        if (!options.toSeconds) {
            std::cerr << "Invalid --to time: " << toToken << "\n";
            return 2;
        }
    }

    return renderWav(session.engine(), options) ? 0 : 1;
}

int execCommands(const juce::StringArray& args) {
    if (args.size() < 3) {
        printUsage(std::cerr);
        return 2;
    }

    RunOptions options;
    options.input = fileFromArg(args[1]);
    options.output = defaultOutputFor(options.input);

    for (int i = 2; i < args.size(); ++i) {
        if (args[i] == "--out") {
            if (++i >= args.size()) {
                printUsage(std::cerr);
                return 2;
            }
            options.output = fileFromArg(args[i]);
        } else if (args[i] == "--dump-json") {
            options.dumpJson = true;
        } else {
            options.execTokens.add(args[i]);
        }
    }

    if (options.execTokens.isEmpty() && !options.dumpJson) {
        printUsage(std::cerr);
        return 2;
    }

    return runCli(options);
}

int bootOnly() {
    HeadlessEngineSession session;
    if (!session.initialize()) {
        std::cerr << session.error() << "\n";
        return 1;
    }

    std::cout << "MAGDA engine booted headless\n";
    return 0;
}

// ============================================================================
// serve mode - persistent engine over a Unix-socket JSON-line protocol (§113 L3)
//
// One request = one JSON object per line:
//   {"op":"exec","commands":["set-tempo","120", ...]}  -> dispatch + save
//   {"op":"dump"}                                      -> full project JSON
//   {"op":"render","wav":"/abs/out.wav",
//    ["from":"8bars"] ["to":"16bars"]}                 -> offline render
//   {"op":"save"}                                      -> save only
// Response = one JSON line: {"ok":true,...} or {"ok":false,"error":...}.
//
// Protocol 2 (see {"op":"describe"}):
//   {"op":"exec",...,"save":false}                     -> dispatch without saving
//   {"op":"notes","clipId":N,"mode":"add"|"replace",
//    "notes":[{"start","length","pitch","velocity"}],"save":bool}
//                                                      -> one undo step, one clip sync
//   {"op":"param_info","deviceId":N,"param":name|index} -> labels, text, value table
//   dump parameters carry text/default/scale/choices/choiceValues/steps, and
//   hosted plugins list their engine-wrapper Dry/Wet pair (owner:"host")
//   {"op":"add_plugin",...} hosts the plugin inside the project model, so it
//   is dumped, saved with its state and reloaded; the ext ops (ext_params,
//   set_ext_param [+index], get/set_ext_state, set_ext_bypass, remove_device)
//   accept model device ids as well as legacy ext ids
//
// Protocol 3 (mixing, arrangement and discovery; `{"op":"commands"}` lists
// every op and exec command with a one-line summary):
//   mixer        {trackId|"master", volume|volumeDb, pan, mute, solo, name}
//   send         {from, to, level|levelDb, remove?}          post-fader sends
//   sidechain    {deviceId, source|null, type:audio|midi}     keyed dynamics
//   add_fx       {trackId|"master", plugin, position:pre|post}
//   automation   {target:{kind,...}, points:[{beat,value,curve}], mode}
//   clip         {action:create|move|resize|duplicate|rename|delete|loop,
//                 overlap:error|replace|shift, ...}
//   render       + {tracks:[ids] | stem:id, masterPlugins}
//   render_stems {dir, tracks?}                               one WAV per channel
//   catalog, commands                                         discovery
//   dump         + master channel, sends, automation lanes, device position
//
// The win: the engine boots once per session instead of once per command,
// turning multi-hundred-command composition pipelines from minutes of boot
// overhead into milliseconds of dispatch.

void sendAll(int conn, const char* data, size_t size) {
    while (size > 0) {
        const auto sent = ::send(conn, data, size, 0);
        if (sent < 0 && errno == EINTR)
            continue;
        if (sent <= 0)
            return;  // client gone; the read loop notices
        data += sent;
        size -= static_cast<size_t>(sent);
    }
}

void sendLine(int conn, const juce::String& text) {
    const char* utf8 = text.toRawUTF8();
    sendAll(conn, utf8, std::strlen(utf8));
    sendAll(conn, "\n", 1);
}

juce::String jsonLine(juce::DynamicObject* obj) {
    return juce::JSON::toString(juce::var(obj), true);
}

juce::String errorJson(const juce::String& message) {
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", false);
    obj->setProperty("error", message);
    return jsonLine(obj);
}

juce::String okJson() {
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    return jsonLine(obj);
}

bool saveCurrentProject() {
    return saveProjectForCli(magda::ProjectManager::getInstance().getCurrentProjectFile());
}

// One line per serve op: what `commands` reports and `describe` lists.
struct ServeOpDoc {
    const char* op;
    const char* summary;
};

const std::vector<ServeOpDoc>& serveOpDocs() {
    static const std::vector<ServeOpDoc> docs = {
        {"ping", "Liveness check."},
        {"describe", "Protocol version, feature flags and op names."},
        {"commands", "Every serve op plus the exec command language (usage and summary)."},
        {"catalog", "Built-in instruments and effects (ids, names, categories) and the number "
                    "of scanned external plugins."},
        {"exec", "{commands:[tokens], save?} run exec commands (see commands)."},
        {"dump", "The project as JSON: tracks (mixer, sends), clips, notes, devices with "
                 "parameter meaning, master channel and automation lanes."},
        {"save", "Save the project."},
        {"notes", "{clipId, mode:add|replace, notes:[{start,length,pitch,velocity}], save?} a "
                  "clip's notes in one undo step (starts relative to the clip)."},
        {"clip", "{action:create|move|resize|duplicate|rename|delete|loop, ...} arrangement "
                 "edits at absolute beats; overlap:error (default)|replace|shift."},
        {"mixer", "{trackId|\"master\", volume|volumeDb, pan, mute, solo, name, save?} channel "
                  "settings."},
        {"send", "{from, to, level|levelDb, remove?, save?} post-fader send to an aux/group "
                 "track."},
        {"sidechain", "{deviceId, source|null, type:audio|midi, save?} key a pre-fader device "
                      "from another track."},
        {"add_fx", "{trackId|\"master\", plugin, name?, position:pre|post, save?} insert a "
                   "built-in effect (see catalog)."},
        {"add_plugin", "{trackId|\"master\", plugin, asInstrument?} host an AU/VST3/LV2 "
                       "plugin inside the project."},
        {"automation", "{target:{kind:volume|pan|send|param, trackId|deviceId, to?, param?}, "
                       "points:[{beat,value,curve?}], mode:replace|add|range|clear, unit?, "
                       "save?} write a lane in real units (dB, -1..1, native range or labels)."},
        {"param_info", "{deviceId, param} one parameter's labels, display text and value "
                       "table."},
        {"render", "{wav, from?, to?, bitDepth?, tracks?:[ids] | stem?:id, masterPlugins?} "
                   "offline render."},
        {"render_stems", "{dir, tracks?:[ids], from?, to?, bitDepth?} one WAV per channel "
                         "(default: every channel that feeds the master)."},
        {"plugins", "Scanned external plugins."},
        {"remove_device", "{deviceId} remove a device."},
        {"ext_params", "{deviceId} a hosted plugin's parameters."},
        {"set_ext_param", "{deviceId, name|index, value 0..1} set a hosted plugin parameter."},
        {"get_ext_state", "{deviceId} a hosted plugin's state (base64)."},
        {"set_ext_state", "{deviceId, state_b64} restore a hosted plugin's state."},
        {"set_ext_bypass", "{deviceId, bypassed} bypass a device."},
        {"shutdown", "Stop the engine."},
    };
    return docs;
}

juce::String commandsJson() {
    juce::Array<juce::var> ops;
    for (const auto& doc : serveOpDocs()) {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("op", juce::String(doc.op));
        obj->setProperty("summary", juce::String(doc.summary));
        ops.add(juce::var(obj));
    }
    juce::Array<juce::var> commands;
    for (const auto& spec : CommandDispatcher::commandSpecs()) {
        if (!spec.showInUsage)
            continue;
        auto* obj = new juce::DynamicObject();
        obj->setProperty("name", juce::String(spec.name));
        obj->setProperty("usage", juce::String(spec.usage));
        obj->setProperty("summary", juce::String(spec.summary != nullptr ? spec.summary : ""));
        commands.add(juce::var(obj));
    }
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("protocol", 3);
    obj->setProperty("ops", ops);
    obj->setProperty("commands", commands);
    return jsonLine(obj);
}

juce::String catalogJson(magda::AudioEngine& engine) {
    juce::Array<juce::var> instruments;
    juce::Array<juce::var> effects;
    auto entry = [](const juce::String& id, const char* name, const char* category,
                    const char* description) {
        auto* obj = new juce::DynamicObject();
        obj->setProperty("id", id);
        obj->setProperty("name", juce::String(name != nullptr ? name : id.toRawUTF8()));
        if (category != nullptr)
            obj->setProperty("category", juce::String(category));
        if (description != nullptr)
            obj->setProperty("description", juce::String(description));
        return obj;
    };
    instruments.add(juce::var(entry("4osc", "4OSC", "Synth",
                                    "Four-oscillator subtractive synth.")));
    juce::StringArray listed;
    for (const auto* spec : magda::daw::audio::compiled::getAllCompiledPluginSpecs()) {
        if (spec == nullptr || spec->pluginId == nullptr)
            continue;
        auto* obj = entry(spec->pluginId, spec->displayName, spec->browserCategory, spec->description);
        (spec->isInstrument ? instruments : effects).add(juce::var(obj));
        listed.add(spec->pluginId);
    }
    for (const auto* spec : magda::daw::audio::getAllInternalPluginSpecs()) {
        if (spec == nullptr || !isInsertableInternalEffect(*spec) || listed.contains(spec->pluginId))
            continue;
        auto* obj = entry(spec->pluginId, spec->displayName, spec->browserCategory, spec->description);
        if (magda::daw::audio::internalPluginHasTag(*spec, "sidechain")) {
            obj->setProperty("sidechain", "midi");
            obj->setProperty("position", "pre");
        }
        effects.add(juce::var(obj));
    }
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("instruments", instruments);
    obj->setProperty("effects", effects);
    obj->setProperty("externalPlugins", static_cast<int>(engine.getKnownPluginTypes().size()));
    return jsonLine(obj);
}

juce::String describeServe() {
    auto* features = new juce::DynamicObject();
    for (const auto* name : {"execSaveFlag", "bulkNotes", "paramLabels", "paramInfo",
                             "pluginPersistence", "extParamIndex",
                             // protocol 3
                             "mixer", "sends", "sidechain", "automation", "clipOps",
                             "clipOverlapPolicy", "clipLoop", "stems", "masterBus", "catalog",
                             "commandList"})
        features->setProperty(name, true);
    juce::Array<juce::var> ops;
    for (const auto& doc : serveOpDocs())
        ops.add(juce::String(doc.op));
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("protocol", 3);
    obj->setProperty("magdaVersion", juce::String(MAGDA_VERSION));
    obj->setProperty("features", juce::var(features));
    obj->setProperty("ops", ops);
    return jsonLine(obj);
}

// {"op":"notes"}: a whole clip's notes in one request -- one undo step and one
// clip re-sync instead of one per note.
juce::String handleNotesOp(const juce::var& request) {
    const auto clipId = intProperty(request, "clipId");
    if (!clipId)
        return errorJson("notes requires numeric clipId");
    const auto mode = request.getProperty("mode", "add").toString();
    if (mode != "add" && mode != "replace")
        return errorJson("notes mode must be \"add\" or \"replace\"");
    const auto notesVar = request.getProperty("notes", juce::var());
    if (!notesVar.isArray())
        return errorJson("notes requires a notes array");
    const auto* clip = magda::ClipManager::getInstance().getClip(*clipId);
    if (clip == nullptr || !clip->isMidi())
        return errorJson("notes: clip " + juce::String(*clipId) + " is not a MIDI clip");

    std::vector<magda::MidiNote> notes;
    notes.reserve(static_cast<size_t>(notesVar.size()));
    for (const auto& n : *notesVar.getArray()) {
        const auto start = n.getProperty("start", juce::var());
        const auto length = n.getProperty("length", juce::var());
        const auto pitch = n.getProperty("pitch", juce::var());
        const auto velocity = n.getProperty("velocity", 100);
        if (!isJsonNumber(start) || !isJsonNumber(length) || !isJsonNumber(pitch) ||
            !isJsonNumber(velocity))
            return errorJson("each note needs numeric start, length, pitch (and velocity)");
        magda::MidiNote note;
        note.startBeat = static_cast<double>(start);
        note.lengthBeats = static_cast<double>(length);
        note.noteNumber = static_cast<int>(static_cast<double>(pitch));
        note.velocity = juce::jlimit(0, 127, static_cast<int>(static_cast<double>(velocity)));
        if (!std::isfinite(note.startBeat) || note.startBeat < 0.0 ||
            !std::isfinite(note.lengthBeats) || note.lengthBeats <= 0.0 || note.noteNumber < 0 ||
            note.noteNumber > 127)
            return errorJson("invalid note (start >= 0, length > 0, pitch 0..127)");
        notes.push_back(note);
    }

    size_t removed = 0;
    size_t added = 0;
    {
        magda::ClipManager::BatchScope notifications;
        magda::CompoundOperationScope undoStep(mode == "replace" ? "Replace MIDI Notes"
                                                                 : "Add MIDI Notes");
        auto& undo = magda::UndoManager::getInstance();
        if (mode == "replace" && !clip->midiNotes.empty()) {
            std::vector<size_t> all(clip->midiNotes.size());
            std::iota(all.begin(), all.end(), size_t{0});
            removed = all.size();
            undo.executeCommand(
                std::make_unique<magda::DeleteMultipleMidiNotesCommand>(*clipId, std::move(all)));
        }
        if (!notes.empty()) {
            auto command = std::make_unique<magda::AddMultipleMidiNotesCommand>(
                *clipId, std::move(notes), "Add MIDI Notes");
            auto* raw = command.get();
            undo.executeCommand(std::move(command));
            added = raw->getInsertedIndices().size();
        }
    }

    if (static_cast<bool>(request.getProperty("save", true)) && !saveCurrentProject())
        return errorJson("failed to save project");

    const auto* after = magda::ClipManager::getInstance().getClip(*clipId);
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("added", static_cast<int>(added));
    obj->setProperty("removed", static_cast<int>(removed));
    obj->setProperty("total", after != nullptr ? static_cast<int>(after->midiNotes.size()) : 0);
    return jsonLine(obj);
}

// {"op":"param_info"}: everything needed to choose a value. Parameters a
// plugin labels without declaring them discrete (Surge's filter type) are
// sampled across their range and returned as choices, each with the value at
// the middle of its label.
juce::String handleParamInfo(const juce::var& request) {
    const auto deviceId = intProperty(request, "deviceId");
    if (!deviceId)
        return errorJson("param_info requires numeric deviceId");
    magda::ChainNodePath path;
    const auto* device = findModelDevice(*deviceId, &path);
    if (device == nullptr)
        return errorJson("unknown device " + juce::String(*deviceId));
    if (device->loadState == magda::DeviceLoadState::Loading)
        return errorJson("device " + juce::String(*deviceId) + " is still loading; retry shortly");
    bool isHost = false;
    const auto* p = resolveDeviceParameter(*device, request.getProperty("param", juce::var()), isHost);
    if (p == nullptr)
        return errorJson("unknown parameter " + request.getProperty("param", juce::var()).toString());

    auto json = parameterToJson(path, *p, isHost);
    auto* obj = json.getDynamicObject();
    obj->setProperty("ok", true);
    if (p->choices.empty() && p->maxValue > p->minValue) {
        constexpr int kSamples = 1000;
        const double lo = p->minValue;
        const double hi = p->maxValue;
        std::vector<juce::String> texts;
        texts.reserve(kSamples + 1);
        bool allLabels = true;
        for (int i = 0; i <= kSamples; ++i) {
            const auto value = static_cast<float>(lo + (hi - lo) * i / kSamples);
            auto text = parameterDisplayText(path, *p, value);
            if (text.isEmpty() || !juce::CharacterFunctions::isLetter(text[0]))
                allLabels = false;
            texts.push_back(std::move(text));
        }
        if (allLabels) {
            juce::Array<juce::var> labels;
            juce::Array<juce::var> values;
            size_t runStart = 0;
            for (size_t i = 1; i <= texts.size(); ++i) {
                if (i == texts.size() || texts[i] != texts[runStart]) {
                    const double middle = (static_cast<double>(runStart) + static_cast<double>(i - 1)) / 2.0;
                    labels.add(texts[runStart]);
                    values.add(lo + (hi - lo) * middle / kSamples);
                    runStart = i;
                }
            }
            if (labels.size() <= 256) {
                obj->setProperty("choices", labels);
                obj->setProperty("choiceValues", values);
                obj->setProperty("steps", labels.size());
                obj->setProperty("valueTable", labels);
            }
        } else {
            juce::Array<juce::var> samples;
            for (int i = 0; i <= kSamples; i += kSamples / 10) {
                auto* sample = new juce::DynamicObject();
                sample->setProperty("value", lo + (hi - lo) * i / kSamples);
                sample->setProperty("text", texts[static_cast<size_t>(i)]);
                samples.add(juce::var(sample));
            }
            obj->setProperty("samples", samples);
        }
    }
    if (!obj->hasProperty("valueTable") && !p->valueTable.empty()) {
        juce::Array<juce::var> table;
        for (size_t i = 0; i < p->valueTable.size() && i < 256; ++i)
            table.add(p->valueTable[i]);
        obj->setProperty("valueTable", table);
    }
    return jsonLine(obj);
}

std::optional<magda::PluginFormat> modelFormatFor(const juce::PluginDescription& desc) {
    if (desc.pluginFormatName == "VST3")
        return magda::PluginFormat::VST3;
    if (desc.pluginFormatName == "AudioUnit" || desc.pluginFormatName == "AU")
        return magda::PluginFormat::AU;
    if (desc.pluginFormatName == "LV2")
        return magda::PluginFormat::LV2;
    return std::nullopt;
}

// {"op":"add_plugin"} (protocol 2): host the plugin inside MAGDA's project
// model -- the same path as dropping it on a track in the app -- so it is in
// `dump`, saved with its native state by the pre-save capture, and reloaded
// with the project. VST3 instances initialise asynchronously; pump the message
// loop until the loader reports the device loaded or failed.
juce::String hostPluginInModel(magda::AudioEngine& engine, int trackId,
                               const juce::PluginDescription& desc, bool asInstrument) {
    const auto format = modelFormatFor(desc);
    if (!format)
        return errorJson("unsupported plugin format: " + desc.pluginFormatName);

    magda::DeviceInfo device;
    device.name = desc.name;
    device.manufacturer = desc.manufacturerName;
    device.pluginId = desc.createIdentifierString();
    device.uniqueId = desc.createIdentifierString();
    device.fileOrIdentifier = desc.fileOrIdentifier;
    device.isInstrument = desc.isInstrument;
    device.deviceType = asInstrument ? magda::DeviceType::Instrument : magda::DeviceType::Effect;
    device.format = *format;

    auto& trackManager = magda::TrackManager::getInstance();
    if (trackManager.getTrack(trackId) == nullptr)
        return errorJson("unknown track " + juce::String(trackId));
    const auto deviceId = asInstrument ? trackManager.addDeviceToTrack(trackId, device, 0)
                                       : trackManager.addDeviceToTrack(trackId, device);
    if (deviceId == magda::INVALID_DEVICE_ID)
        return errorJson("track " + juce::String(trackId) + " cannot host " + desc.name);

    const auto path = trackManager.findDevicePath(deviceId);
    auto* bridge = engine.getAudioBridge();
    auto* messageManager = juce::MessageManager::getInstance();
    const auto deadline = juce::Time::getMillisecondCounter() + 60000;
    const auto graceUntil = juce::Time::getMillisecondCounter() + 2000;
    juce::String error;
    while (true) {
        const auto* info = trackManager.getDeviceInChainByPath(path);
        if (info == nullptr) {
            error = "device disappeared while loading";
            break;
        }
        auto plugin = bridge != nullptr ? bridge->getPlugin(path) : nullptr;
        if (info->loadState == magda::DeviceLoadState::Failed) {
            error = "plugin failed to load";
            if (auto* ext = dynamic_cast<tracktion::engine::ExternalPlugin*>(plugin.get()))
                if (ext->getLoadError().isNotEmpty())
                    error += ": " + ext->getLoadError();
            break;
        }
        auto* ext = dynamic_cast<tracktion::engine::ExternalPlugin*>(plugin.get());
        if (ext != nullptr && !ext->isInitialisingAsync() && ext->getAudioPluginInstance() == nullptr) {
            error = "plugin failed to load";
            if (ext->getLoadError().isNotEmpty())
                error += ": " + ext->getLoadError();
            break;
        }
        // The engine sync runs inside addDeviceToTrack; a device that is not
        // loading and still has no plugin after a grace period failed to load.
        if (plugin == nullptr && info->loadState != magda::DeviceLoadState::Loading &&
            juce::Time::getMillisecondCounter() > graceUntil) {
            error = "plugin failed to load";
            break;
        }
        if (info->loadState == magda::DeviceLoadState::Loaded && plugin != nullptr) {
            auto* obj = new juce::DynamicObject();
            obj->setProperty("ok", true);
            obj->setProperty("device", deviceId);
            obj->setProperty("persistent", true);
            obj->setProperty("parameters", static_cast<int>(info->parameters.size()));
            return jsonLine(obj);
        }
        if (juce::Time::getMillisecondCounter() > deadline) {
            error = "timed out waiting for the plugin to load";
            break;
        }
        messageManager->runDispatchLoopUntil(20);
    }
    if (trackManager.getDeviceInChainByPath(path) != nullptr)
        trackManager.removeDeviceFromChainByPath(path);
    return errorJson(error);
}

// A hosted plugin addressed by a legacy ext id (outside the model) or by a
// model device id.
struct HostedPlugin {
    tracktion::engine::Plugin::Ptr plugin;
    magda::ChainNodePath path;  // valid for model-hosted devices
    bool legacy = false;
    bool loading = false;       // model device whose plugin is still instantiating
};

// Model devices win over legacy ext ids (both count from 1); protocol-2
// clients only create legacy plugins when they ask for them explicitly.
std::optional<HostedPlugin> resolveHostedPlugin(magda::AudioEngine& engine, int id) {
    magda::ChainNodePath path;
    if (const auto* device = findModelDevice(id, &path)) {
        auto* bridge = engine.getAudioBridge();
        auto plugin = bridge != nullptr ? bridge->getPlugin(path) : nullptr;
        const bool loading = device->loadState == magda::DeviceLoadState::Loading;
        if (plugin != nullptr || loading)
            return HostedPlugin{plugin, path, false, loading};
    }
    if (auto* wrapper = asTracktionWrapper(engine))
        if (const auto* entries = wrapper->externalPluginList())
            for (const auto& entry : *entries)
                if (entry.extId == id && entry.plugin != nullptr)
                    return HostedPlugin{entry.plugin, {}, true, false};
    return std::nullopt;
}

juce::String loadingError(int id) {
    return errorJson("device " + juce::String(id) + " is still loading; retry shortly");
}

juce::String handleExtParams(magda::AudioEngine& engine, const juce::var& request) {
    const auto id = intProperty(request, "deviceId");
    if (!id)
        return errorJson("ext_params requires numeric deviceId");
    const auto hosted = resolveHostedPlugin(engine, *id);
    if (!hosted)
        return errorJson("unknown device");
    if (hosted->loading)
        return loadingError(*id);
    juce::Array<juce::var> parameters;
    const auto params = hosted->plugin->getAutomatableParameters();
    for (int i = 0; i < params.size(); ++i) {
        auto* parameter = params[i];
        if (parameter == nullptr)
            continue;
        auto name = parameter->getParameterName();
        if (name.isEmpty())
            name = parameter->getLabel();
        auto* obj = new juce::DynamicObject();
        obj->setProperty("name", name);
        obj->setProperty("index", i);
        obj->setProperty("value", static_cast<double>(parameter->getCurrentNormalisedValue()));
        const auto text = parameter->valueToString(parameter->getCurrentValue()).trim();
        if (text.isNotEmpty())
            obj->setProperty("text", text);
        parameters.add(juce::var(obj));
    }
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("parameters", parameters);
    return jsonLine(obj);
}

juce::String handleSetExtParam(magda::AudioEngine& engine, const juce::var& request) {
    const auto id = intProperty(request, "deviceId");
    const auto valueVar = request.getProperty("value", juce::var());
    if (!id || !isJsonNumber(valueVar))
        return errorJson("set_ext_param requires numeric deviceId and a 0..1 value");
    const auto hosted = resolveHostedPlugin(engine, *id);
    if (!hosted)
        return errorJson("unknown device or parameter");
    if (hosted->loading)
        return loadingError(*id);
    const auto params = hosted->plugin->getAutomatableParameters();
    int target = -1;
    if (const auto index = intProperty(request, "index")) {
        if (*index >= 0 && *index < params.size())
            target = *index;
    } else {
        const auto wanted = request.getProperty("name", {}).toString();
        for (int i = 0; i < params.size() && target < 0; ++i) {
            if (params[i] == nullptr)
                continue;
            auto name = params[i]->getParameterName();
            if (name.isEmpty())
                name = params[i]->getLabel();
            if (name.equalsIgnoreCase(wanted))
                target = i;
        }
    }
    if (target < 0 || params[target] == nullptr)
        return errorJson("unknown device or parameter");
    auto* parameter = params[target];
    const auto clamped = juce::jlimit(0.0f, 1.0f, static_cast<float>(static_cast<double>(valueVar)));
    const auto raw = parameter->valueRange.convertFrom0to1(clamped);
    if (!hosted->legacy) {
        auto& trackManager = magda::TrackManager::getInstance();
        if (auto* device = trackManager.getDeviceInChainByPath(hosted->path);
            device != nullptr && device->findParameterByIndex(target) != nullptr) {
            // Through the model, so the saved project and the dump agree.
            trackManager.setDeviceParameterValue(hosted->path, target, magda::ParameterModelValue{raw});
            return okJson();
        }
    }
    parameter->setParameterFromHost(raw, juce::sendNotificationSync);
    return okJson();
}

juce::String handleGetExtState(magda::AudioEngine& engine, const juce::var& request) {
    const auto id = intProperty(request, "deviceId");
    const auto hosted = id ? resolveHostedPlugin(engine, *id) : std::nullopt;
    if (hosted && hosted->loading)
        return loadingError(*id);
    auto* ext = hosted ? dynamic_cast<tracktion::engine::ExternalPlugin*>(hosted->plugin.get())
                       : nullptr;
    if (ext == nullptr)
        return errorJson("unknown device or no state");
    ext->flushPluginStateToValueTree();
    const auto chunk = ext->state.getProperty(tracktion::engine::IDs::state).toString();
    if (chunk.isEmpty())
        return errorJson("unknown device or no state");
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("state", juce::Base64::toBase64(chunk));
    return jsonLine(obj);
}

juce::String handleSetExtState(magda::AudioEngine& engine, const juce::var& request) {
    const auto id = intProperty(request, "deviceId");
    const auto stateVar = request.getProperty("state_b64", {});
    juce::MemoryOutputStream decoded;
    const bool haveState = stateVar.toString().isNotEmpty() &&
                           juce::Base64::convertFromBase64(decoded, stateVar.toString());
    if (!id || !haveState)
        return errorJson("set_ext_state requires numeric deviceId and non-empty state_b64");
    const auto hosted = resolveHostedPlugin(engine, *id);
    if (hosted && hosted->loading)
        return loadingError(*id);
    auto* ext = hosted ? dynamic_cast<tracktion::engine::ExternalPlugin*>(hosted->plugin.get())
                       : nullptr;
    if (ext == nullptr ||
        !magda::restoreExternalPluginStateDialect(*ext, decoded.getMemoryBlock()))
        return errorJson("unknown device or state rejected by plugin");
    return okJson();
}

juce::String handleSetExtBypass(magda::AudioEngine& engine, const juce::var& request) {
    const auto id = intProperty(request, "deviceId");
    const auto bypassVar = request.getProperty("bypassed", juce::var());
    if (!id || !(bypassVar.isBool() || isJsonNumber(bypassVar)))
        return errorJson("set_ext_bypass requires numeric deviceId and bool bypassed");
    const bool bypassed = bypassVar.isBool() ? static_cast<bool>(bypassVar)
                                             : static_cast<double>(bypassVar) != 0.0;
    const auto hosted = resolveHostedPlugin(engine, *id);
    if (!hosted)
        return errorJson("unknown device");
    if (hosted->legacy) {
        if (auto* wrapper = asTracktionWrapper(engine))
            wrapper->setExternalPluginBypass(*id, bypassed);
    } else {
        magda::TrackManager::getInstance().setDeviceBypassedByPath(hosted->path, bypassed);
    }
    return okJson();
}

juce::String handleRemoveDevice(magda::AudioEngine& engine, const juce::var& request) {
    const auto id = intProperty(request, "deviceId");
    if (!id)
        return errorJson("remove_device requires numeric deviceId");
    bool removed = false;
    magda::ChainNodePath path;
    if (findModelDevice(*id, &path) != nullptr) {
        magda::TrackManager::getInstance().removeDeviceFromChainByPath(path);
        removed = true;
    } else if (auto* wrapper = asTracktionWrapper(engine)) {
        removed = wrapper->removeExternalPlugin(*id);
    }
    if (!removed)
        return errorJson("unknown device");
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("removed", *id);
    return jsonLine(obj);
}

// True while any device in the project is still instantiating (VST3 plugins
// load asynchronously after a project load).
bool anyDeviceLoading() {
    std::function<bool(const std::vector<magda::ChainElement>&)> walk =
        [&walk](const std::vector<magda::ChainElement>& elements) {
            for (const auto& element : elements) {
                if (magda::isDevice(element)) {
                    if (magda::getDevice(element).loadState == magda::DeviceLoadState::Loading)
                        return true;
                } else if (magda::isRack(element)) {
                    for (const auto& chain : magda::getRack(element).chains)
                        if (walk(chain.elements))
                            return true;
                }
            }
            return false;
        };
    for (const auto& track : magda::TrackManager::getInstance().getTracks()) {
        if (walk(track.chain.fxChainElements))
            return true;
        for (const auto& element : track.chain.postFxChainElements)
            if (element.device.loadState == magda::DeviceLoadState::Loading)
                return true;
    }
    return false;
}

// ============================================================================
// Serve protocol 3 ops. Each validates the whole request before changing
// anything, applies it through the project model (so dump, save and the
// engine agree) and saves unless the request says {"save": false}.
// ============================================================================

bool saveRequested(const juce::var& request) {
    const auto save = request.getProperty("save", juce::var());
    return save.isVoid() || static_cast<bool>(save);
}

juce::String okWith(juce::DynamicObject* obj, const juce::var& request) {
    if (saveRequested(request) && !saveCurrentProject()) {
        juce::var discard(obj);  // release the reply we will not send
        return errorJson("failed to save project");
    }
    obj->setProperty("ok", true);
    return jsonLine(obj);
}

// {"op":"mixer","trackId":N|"master","volume"|"volumeDb","pan","mute","solo","name"}
juce::String handleMixerOp(const juce::var& request) {
    auto& trackManager = magda::TrackManager::getInstance();
    const auto trackId = trackProperty(request, "trackId");
    if (!trackId || trackManager.getTrack(*trackId) == nullptr)
        return errorJson("mixer requires a known trackId (or \"master\")");
    const bool isMaster = *trackId == magda::MASTER_TRACK_ID;

    std::optional<double> volume = doubleProperty(request, "volume");
    if (const auto db = doubleProperty(request, "volumeDb")) {
        if (volume)
            return errorJson("give volume (linear) or volumeDb, not both");
        volume = dbToGain(*db);
    }
    if (volume && (*volume < 0.0 || *volume > 2.0 + 1.0e-9))
        return errorJson("volume must be a linear gain in [0, 2] (-inf..+6 dB)");
    const auto pan = doubleProperty(request, "pan");
    if (request.hasProperty("pan") && !pan)
        return errorJson("pan must be a number in [-1, 1]");
    if (pan && (*pan < -1.0 || *pan > 1.0))
        return errorJson("pan must be in [-1, 1]");
    const auto mute = boolProperty(request, "mute");
    if (request.hasProperty("mute") && !mute)
        return errorJson("mute must be true or false");
    const auto solo = boolProperty(request, "solo");
    if (request.hasProperty("solo") && !solo)
        return errorJson("solo must be true or false");
    if (isMaster && solo)
        return errorJson("the master channel has no solo");
    const auto name = request.getProperty("name", juce::var());
    if (!name.isVoid() && (isMaster || !name.isString() || name.toString().trim().isEmpty()))
        return errorJson(isMaster ? "the master channel cannot be renamed"
                                  : "name must be a non-empty string");

    if (volume)
        trackManager.setTrackVolume(*trackId, static_cast<float>(*volume));
    if (pan)
        trackManager.setTrackPan(*trackId, static_cast<float>(*pan));
    if (mute)
        trackManager.setTrackMuted(*trackId, *mute);
    if (solo)
        trackManager.setTrackSoloed(*trackId, *solo);
    if (!name.isVoid())
        trackManager.setTrackName(*trackId, name.toString().trim());

    auto* obj = new juce::DynamicObject();
    obj->setProperty("track", mixerStateJson(*trackId));
    return okWith(obj, request);
}

// {"op":"send","from":N,"to":M,"level"|"levelDb","remove":bool}
juce::String handleSendOp(const juce::var& request) {
    const auto from = trackProperty(request, "from");
    const auto to = trackProperty(request, "to");
    if (!from || !to)
        return errorJson("send requires from and to track ids");
    if (static_cast<bool>(request.getProperty("remove", false))) {
        const auto bus = findSendBus(*from, *to);
        if (!bus)
            return errorJson("track " + trackLabel(*from) + " has no send to track " +
                             trackLabel(*to));
        magda::TrackManager::getInstance().removeSend(*from, *bus);
    } else {
        std::optional<double> level = doubleProperty(request, "level");
        if (const auto db = doubleProperty(request, "levelDb")) {
            if (level)
                return errorJson("give level (linear) or levelDb, not both");
            level = dbToGain(*db);
        }
        juce::String error;
        if (!ensureSend(*from, *to, level, error))
            return errorJson(error);
    }
    auto* obj = new juce::DynamicObject();
    obj->setProperty("track", mixerStateJson(*from));
    return okWith(obj, request);
}

// {"op":"sidechain","deviceId":N,"source":M|null,"type":"audio"|"midi"}
juce::String handleSidechainOp(magda::AudioEngine& engine, const juce::var& request) {
    const auto deviceId = intProperty(request, "deviceId");
    if (!deviceId)
        return errorJson("sidechain requires numeric deviceId");
    if (!request.hasProperty("source"))
        return errorJson("sidechain requires source: a track id, or null to clear the key");
    magda::TrackId source = magda::INVALID_TRACK_ID;
    const auto sourceVar = request.getProperty("source", juce::var());
    if (!sourceVar.isVoid()) {
        const auto parsed = trackProperty(request, "source");
        if (!parsed)
            return errorJson("sidechain source must be a track id (or null)");
        source = *parsed;
    }
    std::optional<magda::SidechainConfig::Type> type;
    const auto typeText = request.getProperty("type", "").toString().trim().toLowerCase();
    if (typeText == "audio")
        type = magda::SidechainConfig::Type::Audio;
    else if (typeText == "midi")
        type = magda::SidechainConfig::Type::MIDI;
    else if (typeText.isNotEmpty())
        return errorJson("sidechain type must be audio or midi");
    juce::String error, applied;
    if (!applySidechain(engine, *deviceId, source, type, error, applied))
        return errorJson(error);
    auto* obj = new juce::DynamicObject();
    obj->setProperty("deviceId", *deviceId);
    auto* sidechain = new juce::DynamicObject();
    sidechain->setProperty("type", applied);
    sidechain->setProperty("source", source == magda::INVALID_TRACK_ID ? juce::var()
                                                                       : juce::var(source));
    obj->setProperty("sidechain", juce::var(sidechain));
    return okWith(obj, request);
}

// {"op":"add_fx","trackId":N|"master","plugin":id,"name":str,"position":"pre"|"post"}
juce::String handleAddFxOp(const juce::var& request) {
    auto& trackManager = magda::TrackManager::getInstance();
    const auto trackId = trackProperty(request, "trackId");
    if (!trackId || trackManager.getTrack(*trackId) == nullptr)
        return errorJson("add_fx requires a known trackId (or \"master\")");
    const auto plugin = request.getProperty("plugin", "").toString();
    const auto effect = resolveInternalEffect(plugin);
    if (!effect)
        return errorJson("unknown built-in effect \"" + plugin + "\" (known: " + knownEffectHint() +
                         "; host AU/VST3 plugins with add_plugin)");
    const auto position = request.getProperty("position", "pre").toString().trim().toLowerCase();
    if (position != "pre" && position != "post")
        return errorJson("position must be pre (main chain, before the fader) or post");
    juce::String error;
    const auto name = request.getProperty("name", "").toString().trim();
    const auto deviceId = addInternalEffect(*trackId, *effect, name.isNotEmpty() ? name : plugin.trim(),
                                            position == "pre", error);
    if (deviceId == magda::INVALID_DEVICE_ID)
        return errorJson(error);
    auto* obj = new juce::DynamicObject();
    obj->setProperty("device", deviceId);
    obj->setProperty("pluginId", effect->pluginId);
    obj->setProperty("position", position);
    if (const auto* device = findModelDevice(deviceId))
        obj->setProperty("parameters", static_cast<int>(device->parameters.size()));
    return okWith(obj, request);
}

juce::String handleAutomationOp(const juce::var& request) {
    juce::DynamicObject::Ptr result = new juce::DynamicObject();
    const auto error = applyAutomationRequest(request, *result);
    if (error.isNotEmpty())
        return errorJson(error);
    return okWith(result.get(), request);
}

// {"op":"clip","action":...}: arrangement edits at absolute beats.
juce::String handleClipOp(const juce::var& request) {
    auto& clipManager = magda::ClipManager::getInstance();
    const auto action = request.getProperty("action", "").toString().trim().toLowerCase();
    const auto overlap = request.getProperty("overlap", "error").toString().trim().toLowerCase();
    if (overlap != "error" && overlap != "replace" && overlap != "shift")
        return errorJson("overlap must be error, replace or shift");
    juce::String error;
    magda::ClipId clipId = magda::INVALID_CLIP_ID;
    bool shifted = false;

    if (action == "create") {
        const auto trackId = trackProperty(request, "trackId");
        const auto start = doubleProperty(request, "start");
        const auto length = doubleProperty(request, "length");
        if (!trackId || !start || !length)
            return errorJson("clip create requires trackId, start and length (beats)");
        if (!clipTrackUsable(*trackId, error))
            return errorJson(error);
        if (*start < 0.0 || *length <= 0.0)
            return errorJson("clip create requires start >= 0 and length > 0");
        std::vector<magda::MidiNote> notes;
        if (request.hasProperty("notes") &&
            !parseClipNotes(request.getProperty("notes", juce::var()), *length, notes, error))
            return errorJson(error);
        if (overlap == "error") {
            const auto hits = overlappingClips(*trackId, *start, *length, magda::INVALID_CLIP_ID);
            if (!hits.empty())
                return errorJson("the clip would overlap " + describeClipSpans(hits) +
                                 "; pick a free range, or overlap replace (trims what is "
                                 "underneath) or shift (moves the new clip later)");
        }
        clipId = clipManager.createMidiClipBeats(
            *trackId, *start, *length, magda::ClipView::Arrangement,
            overlap == "replace" ? magda::ClipOverlapPolicy::ResolveOverlaps
                                 : magda::ClipOverlapPolicy::PreserveExisting);
        if (clipId == magda::INVALID_CLIP_ID || clipManager.getClip(clipId) == nullptr)
            return errorJson("the engine could not create the clip");
        shifted = std::abs(clipManager.getClip(clipId)->placement.startBeat - *start) > 1.0e-9;
        const auto name = request.getProperty("name", "").toString().trim();
        if (name.isNotEmpty())
            clipManager.setClipName(clipId, name);
        addNotesToClip(clipId, std::move(notes));
    } else {
        const auto requested = intProperty(request, "clipId");
        if (!requested || arrangementClip(*requested, error) == nullptr)
            return errorJson(requested ? error : juce::String("clip actions need a numeric clipId"));
        clipId = *requested;
        std::optional<magda::TrackId> track;
        if (request.hasProperty("trackId")) {
            track = trackProperty(request, "trackId");
            if (!track)
                return errorJson("trackId must be a track id");
        }
        if (action == "move") {
            const auto start = doubleProperty(request, "start");
            if (!start && !track)
                return errorJson("clip move requires start and/or trackId");
            const double at = start.value_or(clipManager.getClip(clipId)->placement.startBeat);
            if (!moveClipChecked(clipId, track, at, overlap == "replace", error))
                return errorJson(error);
        } else if (action == "resize") {
            const auto length = doubleProperty(request, "length");
            if (!length)
                return errorJson("clip resize requires length (beats)");
            if (!resizeClipChecked(clipId, *length, overlap == "replace", error))
                return errorJson(error);
        } else if (action == "duplicate") {
            const auto start = doubleProperty(request, "start");
            if (!start)
                return errorJson("clip duplicate requires start (beats)");
            const auto copy = duplicateClipChecked(clipId, track, *start, overlap, error);
            if (copy == magda::INVALID_CLIP_ID)
                return errorJson(error);
            shifted = std::abs(clipManager.getClip(copy)->placement.startBeat - *start) > 1.0e-9;
            clipId = copy;
            const auto name = request.getProperty("name", "").toString().trim();
            if (name.isNotEmpty())
                clipManager.setClipName(clipId, name);
        } else if (action == "rename") {
            const auto name = request.getProperty("name", "").toString().trim();
            if (name.isEmpty())
                return errorJson("clip rename requires a non-empty name");
            clipManager.setClipName(clipId, name);
        } else if (action == "delete") {
            clipManager.deleteClip(clipId);
            auto* obj = new juce::DynamicObject();
            obj->setProperty("deleted", clipId);
            return okWith(obj, request);
        } else if (action == "loop") {
            const auto enabled = boolProperty(request, "enabled").value_or(true);
            if (!setClipLoopChecked(clipId, enabled, doubleProperty(request, "loopLength"),
                                    doubleProperty(request, "length"), overlap == "replace", error))
                return errorJson(error);
        } else {
            return errorJson("clip action must be create, move, resize, duplicate, rename, delete "
                             "or loop");
        }
    }

    const auto* clip = clipManager.getClip(clipId);
    if (clip == nullptr)
        return errorJson("the clip disappeared during the edit");
    auto json = clipToJson(*clip);
    if (auto* clipObj = json.getDynamicObject())
        clipObj->setProperty("trackId", clip->trackId);
    auto* obj = new juce::DynamicObject();
    obj->setProperty("clip", json);
    if (shifted)
        obj->setProperty("shifted", true);
    return okWith(obj, request);
}

// Fills the range/format/track options of a render request.
juce::String parseRenderRequest(const juce::var& request, RenderOptions& options) {
    auto& trackManager = magda::TrackManager::getInstance();
    const auto bitDepth = request.getProperty("bitDepth", juce::var());
    if (bitDepth.isInt() || bitDepth.isDouble())
        options.bitDepth = parsePositiveInt(juce::String(bitDepth.toString()));
    const auto& info = magda::ProjectManager::getInstance().getCurrentProjectInfo();
    const auto fromToken = request.getProperty("from", "").toString();
    const auto toToken = request.getProperty("to", "").toString();
    if (fromToken.isNotEmpty()) {
        options.fromSeconds = parseRenderTime(fromToken, info);
        if (!options.fromSeconds)
            return "invalid from time: " + fromToken;
    }
    if (toToken.isNotEmpty()) {
        options.toSeconds = parseRenderTime(toToken, info);
        if (!options.toSeconds)
            return "invalid to time: " + toToken;
    }
    const auto tracks = request.getProperty("tracks", juce::var());
    if (tracks.isArray()) {
        for (const auto& t : *tracks.getArray()) {
            const auto id = isJsonNumber(t) ? std::optional<magda::TrackId>(
                                                  static_cast<int>(static_cast<double>(t)))
                                            : parseTrackRef(t.toString());
            if (!id || *id == magda::MASTER_TRACK_ID || trackManager.getTrack(*id) == nullptr)
                return "unknown track in tracks: " + t.toString();
            options.trackIds.push_back(*id);
        }
    }
    if (request.hasProperty("stem")) {
        const auto stem = trackProperty(request, "stem");
        if (!stem || *stem == magda::MASTER_TRACK_ID || trackManager.getTrack(*stem) == nullptr)
            return "stem must be a known track id";
        if (!options.trackIds.empty())
            return "pass tracks or stem, not both";
        options.stemTrack = *stem;
    }
    if (const auto masterPlugins = boolProperty(request, "masterPlugins"))
        options.useMasterPlugins = *masterPlugins;
    return {};
}

// {"op":"render_stems","dir":"/abs/dir","tracks":[...],"from","to","bitDepth"}
juce::String handleRenderStems(magda::AudioEngine& engine, const juce::var& request) {
    const auto dirText = request.getProperty("dir", "").toString();
    if (dirText.isEmpty() || !juce::File::isAbsolutePath(dirText))
        return errorJson("render_stems requires an absolute dir");
    const juce::File dir(dirText);
    if (!dir.createDirectory())
        return errorJson("cannot create " + dirText);
    RenderOptions base;
    const auto parseError = parseRenderRequest(request, base);
    if (parseError.isNotEmpty())
        return errorJson(parseError);
    if (base.stemTrack)
        return errorJson("render_stems takes tracks, not stem");
    auto& trackManager = magda::TrackManager::getInstance();
    auto stems = base.trackIds;
    if (stems.empty())
        for (const auto& track : trackManager.getTracks())
            if (track.type != magda::TrackType::Chord &&
                !track.audioOutputDevice.startsWith("track:"))
                stems.push_back(track.id);
    juce::Array<juce::var> results;
    int number = 0;
    for (const auto trackId : stems) {
        const auto* track = trackManager.getTrack(trackId);
        if (track == nullptr)
            continue;
        const auto name = track->name;
        const bool muted = track->muted;
        auto options = base;
        options.trackIds.clear();
        options.stemTrack = trackId;
        const auto fileName = juce::File::createLegalFileName(
            name.isNotEmpty() ? name : "track-" + juce::String(trackId)).replaceCharacter(' ', '_');
        options.wavOutput = dir.getChildFile(juce::String(++number).paddedLeft('0', 2) + "-" +
                                             fileName + ".wav");
        if (!renderWav(engine, options))
            return errorJson("stem render failed for track " + juce::String(trackId));
        auto* entry = new juce::DynamicObject();
        entry->setProperty("trackId", trackId);
        entry->setProperty("name", name);
        entry->setProperty("file", options.wavOutput.getFullPathName());
        if (muted)
            entry->setProperty("muted", true);
        results.add(juce::var(entry));
    }
    auto* obj = new juce::DynamicObject();
    obj->setProperty("ok", true);
    obj->setProperty("stems", results);
    return jsonLine(obj);
}

int serveSession(const juce::StringArray& args) {
    // NOTE: like every other handler here, args[0] is still the command
    // token ("serve"); the project path lives at args[1].
    if (args.size() < 2) {
        printUsage(std::cerr);
        return 2;
    }

    juce::File input = fileFromArg(args[1]);
    juce::String sockPath;
    for (int i = 2; i < args.size(); ++i) {
        if (args[i] == "--sock") {
            if (++i >= args.size()) {
                std::cerr << "serve requires a path after --sock\n";
                return 2;
            }
            sockPath = args[i];
        } else {
            printUsage(std::cerr);
            return 2;
        }
    }
    if (sockPath.isEmpty())
        sockPath = input.getParentDirectory().getFullPathName() + "/"
                   + input.getFileNameWithoutExtension() + ".sock";

    HeadlessEngineSession session;
    if (!session.initialize()) {
        std::cerr << session.error() << "\n";
        return 1;
    }
    if (!loadProjectForCli(input, session)) {
        return 1;
    }

    // A client that disconnects mid-reply must not kill the engine.
    ::signal(SIGPIPE, SIG_IGN);

    // Answer requests only once hosted plugins restored with the project are
    // live, so a render or parameter query right after boot sees them.
    {
        auto* messageManager = juce::MessageManager::getInstance();
        const auto settle = juce::Time::getMillisecondCounter() + 250;
        const auto deadline = juce::Time::getMillisecondCounter() + 60000;
        while (juce::Time::getMillisecondCounter() < settle ||
               (anyDeviceLoading() && juce::Time::getMillisecondCounter() < deadline))
            messageManager->runDispatchLoopUntil(20);
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    const auto sockStd = sockPath.toStdString();
    if (sockStd.size() >= sizeof(addr.sun_path)) {
        std::cerr << "socket path too long: " << sockPath << "\n";
        return 2;
    }
    std::strncpy(addr.sun_path, sockStd.c_str(), sizeof(addr.sun_path) - 1);
    unlink(sockStd.c_str());  // stale socket from a crashed session
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        std::cerr << "socket() failed: " << std::strerror(errno) << "\n";
        return 1;
    }
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "bind(" << sockPath << ") failed: " << std::strerror(errno) << "\n";
        ::close(fd);
        return 1;
    }
    if (::listen(fd, 4) != 0) {
        std::cerr << "listen failed: " << std::strerror(errno) << "\n";
        ::close(fd);
        return 1;
    }
    std::cout.flush();
    std::cerr << "[serve] listening on " << sockPath << "\n";
    std::cout << "SERVING " << sockPath << "\n";
    std::cout.flush();

    CommandDispatcher dispatcher(session.engine());
    bool running = true;

    // CRITICAL: this function runs on the main thread, which hosts the JUCE
    // message loop. Blocking forever in accept()/recv() starves it and every
    // engine operation deadlocks. Poll with short timeouts and pump messages.
    auto* messageManager = juce::MessageManager::getInstance();
    auto pumpMessages = [messageManager]() {
        messageManager->runDispatchLoopUntil(10);
    };

    while (running) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        timeval tv {0, 20000};  // 20 ms
        const int ready = ::select(fd + 1, &rfds, nullptr, nullptr, &tv);
        pumpMessages();
        if (ready <= 0 || !FD_ISSET(fd, &rfds))
            continue;  // nothing to accept yet - keep pumping

        const int conn = ::accept(fd, nullptr, nullptr);
        if (conn < 0)
            continue;
        std::string line;
        char buf[4096];
        bool clientGone = false;
        while (running && !clientGone) {
            // Read one newline-delimited request, pumping while we wait.
            line.clear();  // fresh buffer per request
            ssize_t n = 0;
            bool gotLine = false;
            while (!gotLine && !clientGone) {
                fd_set cfds;
                FD_ZERO(&cfds);
                FD_SET(conn, &cfds);
                timeval ctv {0, 10000};  // 10 ms
                const int cready =
                    ::select(conn + 1, &cfds, nullptr, nullptr, &ctv);
                if (cready > 0 && FD_ISSET(conn, &cfds)) {
                    n = ::recv(conn, buf, sizeof(buf), 0);
                    if (n <= 0) {  // client disconnected
                        clientGone = true;
                        break;
                    }
                    for (ssize_t i = 0; i < n; ++i) {
                        if (buf[i] == '\n') {
                            gotLine = true;
                            break;
                        }
                        line.push_back(buf[i]);
                    }
                } else {
                    pumpMessages();  // keep the engine alive while idling
                }
            }
            if (clientGone)
                break;
            if (!gotLine)
                continue;
            pumpMessages();  // let queued work settle before dispatching

            juce::var request;
            const auto parseErr = juce::JSON::parse(juce::String(line), request);
            juce::String response;
            if (request.isVoid()) {
                response = "{\"ok\":false,\"error\":\"bad json request\"}";
            } else {
                const auto op = request.getProperty("op", "").toString();
                auto& engine = session.engine();
                if (op == "ping") {
                    response = "{\"ok\":true,\"pong\":true}";
                } else if (op == "shutdown") {
                    response = "{\"ok\":true,\"bye\":true}";
                    running = false;
                } else if (op == "describe") {
                    response = describeServe();
                } else if (op == "notes") {
                    response = handleNotesOp(request);
                } else if (op == "param_info") {
                    response = handleParamInfo(request);
                } else if (op == "commands") {
                    response = commandsJson();
                } else if (op == "catalog") {
                    response = catalogJson(engine);
                } else if (op == "mixer") {
                    response = handleMixerOp(request);
                } else if (op == "send") {
                    response = handleSendOp(request);
                } else if (op == "sidechain") {
                    response = handleSidechainOp(engine, request);
                } else if (op == "add_fx") {
                    response = handleAddFxOp(request);
                } else if (op == "automation") {
                    response = handleAutomationOp(request);
                } else if (op == "clip") {
                    response = handleClipOp(request);
                } else if (op == "render_stems") {
                    response = handleRenderStems(engine, request);
                } else if (op == "exec" || op == "dump" || op == "render" || op == "save") {
                    CommandResult result;
                    juce::String extra;
                    size_t cursor = 0;
                    if (op == "exec") {
                        const auto cmdsVar = request.getProperty("commands", {});
                        juce::StringArray tokens;
                        if (cmdsVar.isArray())
                            for (const auto& v : *cmdsVar.getArray())
                                tokens.add(v.toString());
                        if (tokens.isEmpty()) {
                            result = makeFail("exec requires a non-empty commands array");
                        } else {
                            // Dispatch every token, mirroring executeCommandTokens.
                            bool allOk = true;
                            while (cursor < static_cast<size_t>(tokens.size())) {
                                auto r = dispatcher.execute(tokens, cursor);
                                if (!r.ok) {
                                    result = r;
                                    allOk = false;
                                    break;
                                }
                            }
                            // Protocol 2: {"save":false} skips the per-request
                            // save; clients save once with {"op":"save"}.
                            const auto saveVar = request.getProperty("save", juce::var());
                            const bool doSave = saveVar.isVoid() || static_cast<bool>(saveVar);
                            if (allOk && doSave && !saveCurrentProject()) {
                                result = makeFail("failed to save project");
                            }
                        }
                    } else if (op == "dump") {
                        sendLine(conn, dumpProjectJson(engine.getMagdaApi()));
                        continue;  // already answered
                    } else if (op == "save") {
                        result = saveCurrentProject() ? CommandResult{}
                                                      : makeFail("failed to save project");
                    } else {  // render
                        RenderOptions ro;
                        const auto wav = request.getProperty("wav", "").toString();
                        if (wav.isEmpty() || !juce::File::isAbsolutePath(wav)) {
                            result = makeFail("render requires an absolute wav path");
                        } else {
                            ro.wavOutput = juce::File(wav);
                            const auto parseError = parseRenderRequest(request, ro);
                            if (parseError.isNotEmpty())
                                result = makeFail(parseError);
                            else if (!renderWav(engine, ro))
                                result = makeFail("render failed");
                        }
                    }
                    response = result.ok ? okJson() : errorJson(result.error);
                } else if (op == "plugins") {
                    auto* obj = new juce::DynamicObject();
                    juce::Array<juce::var> plugins;
                    if (auto* wrapper = asTracktionWrapper(engine)) {
                        for (const auto& desc : wrapper->getKnownPluginTypes()) {
                            auto* entry = new juce::DynamicObject();
                            entry->setProperty("id", desc.createIdentifierString());
                            entry->setProperty("name", desc.name);
                            entry->setProperty("format", desc.pluginFormatName);
                            entry->setProperty("isInstrument", desc.isInstrument);
                            entry->setProperty("manufacturer", desc.manufacturerName);
                            plugins.add(juce::var(entry));
                        }
                    }
                    obj->setProperty("ok", true);
                    obj->setProperty("plugins", plugins);
                    response = jsonLine(obj);
                } else if (op == "add_plugin") {
                    const auto trackNum = trackProperty(request, "trackId");
                    const auto pluginVar = request.getProperty("plugin", {});
                    const bool asInstrument =
                        static_cast<bool>(request.getProperty("asInstrument", false));
                    auto* wrapper = asTracktionWrapper(engine);
                    auto description = pluginVar.toString().isEmpty()
                                           ? std::nullopt
                                           : findExternalPluginDescription(
                                                 engine, pluginVar.toString());
                    if (wrapper == nullptr) {
                        response = errorJson("external plugins require the Tracktion engine");
                    } else if (!trackNum) {
                        response = errorJson("add_plugin requires a trackId (or \"master\")");
                    } else if (!description) {
                        response = errorJson("unknown plugin: " + pluginVar.toString());
                    } else if (!static_cast<bool>(request.getProperty("legacy", false))) {
                        response = hostPluginInModel(engine, *trackNum, *description, asInstrument);
                    } else {
                        // Legacy: hosted beside the model (not saved with the project).
                        const int extId = wrapper->addExternalPlugin(
                            std::to_string(*trackNum), *description, asInstrument);
                        if (extId > 0) {
                            auto* obj = new juce::DynamicObject();
                            obj->setProperty("ok", true);
                            obj->setProperty("device", extId);
                            obj->setProperty("persistent", false);
                            response = jsonLine(obj);
                        } else {
                            response = errorJson("failed to instantiate plugin");
                        }
                    }
                } else if (op == "remove_device") {
                    response = handleRemoveDevice(engine, request);
                } else if (op == "ext_params") {
                    response = handleExtParams(engine, request);
                } else if (op == "set_ext_param") {
                    response = handleSetExtParam(engine, request);
                } else if (op == "get_ext_state") {
                    response = handleGetExtState(engine, request);
                } else if (op == "set_ext_state") {
                    response = handleSetExtState(engine, request);
                } else if (op == "set_ext_bypass") {
                    response = handleSetExtBypass(engine, request);
                } else {
                    response = errorJson("unknown op: " + op);
                }
            }
            sendLine(conn, response);
        }
        ::close(conn);
    }
    ::close(fd);
    unlink(sockStd.c_str());
    std::cerr << "[serve] session ended\n";
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    juce::ScopedJuceInitialiser_GUI juceInit;

    juce::StringArray args;
    for (int i = 0; i < argc; ++i)
        args.add(juce::String(argv[i]));

    if (args.size() < 2 || args[1] == "--help" || args[1] == "-h") {
        printUsage(args.size() < 2 ? std::cerr : std::cout);
        return args.size() < 2 ? 2 : 0;
    }

    const auto command = args[1];
    args.remove(0);

    if (command == "boot")
        return bootOnly();
    if (command == "init")
        return initProject(args);
    if (command == "run")
        return runRoundTrip(args);
    if (command == "exec")
        return execCommands(args);
    if (command == "render")
        return renderProject(args);
    if (command == "serve")
        return serveSession(args);
    if (command == "commands") {
        // Machine-readable discovery without booting the engine.
        if (args.contains("--json"))
            std::cout << commandsJson() << "\n";
        else
            printUsage(std::cout);
        return 0;
    }

    std::cerr << "Unknown command: " << command << "\n";
    printUsage(std::cerr);
    return 2;
}
