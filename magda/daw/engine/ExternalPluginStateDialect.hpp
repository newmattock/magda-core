#pragma once

// Harness state dialect for hosted external plugins (CLI / serve).
//
// Shared by the legacy extId path (TracktionEngineWrapper::setExternalPluginState)
// and devices hosted inside MAGDA's project model (serve `set_ext_state`), so a
// sampler loaded either way understands the same payloads.

#include <tracktion_engine/tracktion_engine.h>

#include "../audio/plugin_manager/ExternalPluginStateUtil.hpp"

#include <cstdio>
#include <memory>

namespace magda {

/**
 * Restore a hosted external plugin's state from a harness payload.
 *
 *  - `<form><param name="file" value="..."/></form>` is translated into
 *    sfizz's native chunk;
 *  - a Tracktion state string (what serve `get_ext_state` returns) is applied
 *    as-is, so get -> set round-trips;
 *  - anything else is treated as the plugin's raw native state bytes.
 *
 * The chunk is applied with MAGDA's own helper: it is published on the
 * plugin's state (so an instance that is still loading picks it up) and TE's
 * parameter cache is refreshed (so the next graph build does not revert it).
 */
inline bool restoreExternalPluginStateDialect(tracktion::engine::ExternalPlugin& plugin,
                                              const juce::MemoryBlock& payload) {
    namespace te = tracktion::engine;
    auto* ext = &plugin;
    const auto state = payload.toString();
    // MAGDA harnesses address sampler instruments with a small XML
    // dialect: <form><param name="file" value="..."/></form>.
    // Translate the known keys into the plugin's NATIVE chunk and
    // restore through TE, exactly like a host preset would.
    juce::String sfzPath, volumeParam;
    if (state.contains("<form>") && state.contains("name=\"file\"")) {
        std::unique_ptr<juce::XmlElement> xml(
            juce::XmlDocument::parse(state));
        // NOTE: parse() returns the <form> ROOT element itself;
        // its children ARE the <param> entries.
        juce::XmlElement* form =
            (xml != nullptr && xml->getTagName() == "form")
                ? xml.get()
                : (xml != nullptr ? xml->getChildByName("form")
                                  : nullptr);
        if (form != nullptr) {
            for (int i = 0; i < form->getNumChildElements(); ++i) {
                auto* param = form->getChildElement(i);
                if (param == nullptr
                    || param->getTagName() != "param")
                    continue;
                auto name = param->getStringAttribute("name");
                auto value = param->getStringAttribute("value");
                if (name == "file" && value.isNotEmpty())
                    sfzPath = value;
                else if (name.equalsIgnoreCase("volume"))
                    volumeParam = value;
            }
        }
    }

    juce::String chunkString;
    if (sfzPath.isNotEmpty()) {
        // Build the plugin's native state chunk. sfizz 1.2.x stores
        // a little-endian struct: u64 version, str8-prefixed
        // platform-agnostic sfz path, f32 volume, i32 numVoices,
        // i32 oversamplingLog2, i32 preloadSize, then scala/quality/
        // keyswitch fields and an empty CC map (see SfizzVstState,
        // currentStateVersion 5). Other plugins keep their previous
        // behaviour: the raw state is passed through as-is.
        const bool isSfizz =
            ext->desc.manufacturerName.containsIgnoreCase("SFZTools")
            || ext->desc.name.containsIgnoreCase("sfizz");
        if (isSfizz) {
            std::fprintf(stderr,
                         "[magda] translating form-dialect state "
                         "into sfizz native chunk (sfz=%s)\n",
                         sfzPath.toRawUTF8());
            juce::MemoryBlock chunk;
            const juce::uint64 version = 5;
            chunk.append(&version, sizeof(version));
            auto writeStr8 = [&chunk](const juce::String& s) {
                const juce::uint32 len =
                    (juce::uint32) s.getNumBytesAsUTF8();
                chunk.append(&len, sizeof(len));
                chunk.append(s.toRawUTF8(), len);
            };
            writeStr8(sfzPath);                       // sfzFile
            float volume = 0.0f;                      // dB, default
            if (volumeParam.isNotEmpty())
                volume = volumeParam.getFloatValue();
            chunk.append(&volume, sizeof(volume));
            const juce::int32 numVoices = 64;
            chunk.append(&numVoices, sizeof(numVoices));
            const juce::int32 oversamplingLog2 = 0;
            chunk.append(&oversamplingLog2, sizeof(oversamplingLog2));
            const juce::int32 preloadSize = 8192;
            chunk.append(&preloadSize, sizeof(preloadSize));
            const char* emptyStr = "";                // scalaFile
            const juce::uint32 emptyLen = 0;
            chunk.append(&emptyLen, sizeof(emptyLen));
            chunk.append(emptyStr, 0);
            const juce::int32 scalaRootKey = 60;
            chunk.append(&scalaRootKey, sizeof(scalaRootKey));
            const float tuningFrequency = 440.0f;
            chunk.append(&tuningFrequency,
                         sizeof(tuningFrequency));
            const float stretchedTuning = 0.0f;
            chunk.append(&stretchedTuning,
                         sizeof(stretchedTuning));
            const juce::int32 sampleQuality = 2;
            chunk.append(&sampleQuality, sizeof(sampleQuality));
            const juce::int32 oscillatorQuality = 1;
            chunk.append(&oscillatorQuality,
                         sizeof(oscillatorQuality));
            const juce::int32 freewheelingSampleQuality = 10;
            chunk.append(&freewheelingSampleQuality,
                         sizeof(freewheelingSampleQuality));
            const juce::int32 freewheelingOscillatorQuality = 3;
            chunk.append(&freewheelingOscillatorQuality,
                         sizeof(freewheelingOscillatorQuality));
            const juce::uint8 sustainCancelsRelease = 0;
            chunk.append(&sustainCancelsRelease,
                         sizeof(sustainCancelsRelease));
            const juce::int32 lastKeyswitch = -1;
            chunk.append(&lastKeyswitch, sizeof(lastKeyswitch));
            const juce::uint32 ccCount = 0;           // no CC overrides
            chunk.append(&ccCount, sizeof(ccCount));

            chunkString = chunk.toBase64Encoding();
        } else {
            chunkString = payload.toBase64Encoding();
        }
    } else {
        // Not our dialect. A Tracktion state string ("<size>.<data>", as
        // get_ext_state hands out) is already in the form TE stores; anything
        // else is the plugin's native state bytes.
        juce::MemoryBlock probe;
        const bool teChunk = state.containsOnly("0123456789.+/=_-abcdefghijklmnopqrstuvwxyz"
                                                "ABCDEFGHIJKLMNOPQRSTUVWXYZ") &&
                             state.upToFirstOccurrenceOf(".", false, false)
                                 .containsOnly("0123456789") &&
                             state.contains(".") && probe.fromBase64Encoding(state);
        chunkString = teChunk ? state : payload.toBase64Encoding();
    }
    applyExternalPluginChunk(ext, chunkString);
    juce::ignoreUnused(te::IDs::state);
    return true;
}

}  // namespace magda
