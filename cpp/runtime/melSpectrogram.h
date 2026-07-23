/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "audioLoader.h"
#include "common/tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
namespace audio
{

//! Type of log applied to mel-power.
enum class LogType
{
    kLog10, //!< Whisper-style log10 with a 1e-10 floor.
    kLn,    //!< Natural log (Parakeet).
};

//! Floor strategy used right before the log.
//!
//! HF FE call sites:
//!  - Whisper:  ``np.maximum(mel_floor=1e-10, mel)`` -> kMax
//!  - Parakeet: ``torch.log(mel + 2**-24)``          -> kAdd
enum class LogFloorMode
{
    kMax, //!< ``log(max(mel, logFloor))``
    kAdd, //!< ``log(mel + logFloor)``
};

//! Mel scale convention. HF FE picks one per model.
enum class MelScale
{
    kHtk,    //!< ``2595 log10(1 + f/700)`` — librosa(htk=True).
    kSlaney, //!< Auditory toolbox / librosa default (htk=False). Used by HF
             //!< WhisperFeatureExtractor and ParakeetFeatureExtractor.
    kKaldi,  //!< ``1127 ln(1 + f/700)`` — Kaldi-style; pair with
             //!< ``triangulariseInMelSpace=true``.
};

//! Mel filter normalization. Slaney area-norm matches HF defaults for
//! Whisper / Parakeet.
enum class MelNorm
{
    kNone,
    kSlaney,
};

//! Window function applied to each frame.
enum class WindowType
{
    kHannPeriodic,     //!< ``torch.hann_window(N, periodic=True)``; HF Whisper default.
    kHannSymmetric,    //!< ``torch.hann_window(N, periodic=False)``; HF Parakeet.
    kHammingPeriodic,  //!< ``torch.hamming_window(N, periodic=True)``.
    kHammingSymmetric, //!< ``torch.hamming_window(N, periodic=False)``.
};

//! Output layout for the mel tensor.
enum class MelLayout
{
    kMelTime, //!< ``[n_mel, T]`` — Whisper convention.
    kTimeMel, //!< ``[T, n_mel]`` — Parakeet convention.
};

//! Optional post-processing applied after log.
//!
//! Whisper:
//!   ``log_spec = max(log_spec, log_spec.max() - 8.0)``
//!   ``log_spec = (log_spec + 4.0) / 4.0``
//!
//! Parakeet: subtract per-mel mean, divide by per-mel std (over valid T).
enum class PostNormalize
{
    kNone,
    kWhisperClamp,
    kPerFeatureMeanStd,
};

//! How successive frames slice the waveform.
//!
//! HF transformers' ``audio_utils.spectrogram(center=True, pad_mode="reflect")``
//! reflects ``n_fft / 2`` samples on each side before framing, which centres
//! frame ``t`` at sample ``t * hop``. Our left-aligned legacy frames frame
//! ``t`` at samples ``[t*hop, t*hop + n_fft)``.
enum class FramePadding
{
    kLeftAlignedZero, //!< Legacy: frame[t] = samples[t*hop .. t*hop+nFFT). Zero-pad tail.
    kCenterReflect,   //!< HF Whisper: reflect-pad nFFT/2 each side, then frame
                      //!< (audio_utils.spectrogram(center=True, pad_mode="reflect")).
    kCenterZero,      //!< HF Parakeet: zero-pad nFFT/2 each side, then frame
                      //!< (torch.stft(pad_mode="constant"), default for HF Parakeet).
    kSemicausalZero,  //!< HF Gemma4 (sl.STFT time_padding='semicausal'): zero-pad
                      //!< winLength/2 on the left only; frames unfold with size
                      //!< winLength+1 (the trailing sample is dropped before the
                      //!< window when preemphasis is off).
};

//! Time-axis padding policy.
//!
//! Whisper extracts mel over the full audio without padding to a fixed T
//! (the model handles up to ``chunk_size=30s -> T=3000``). Engines that
//! bake static seq constants into the ONNX trace can use ``kStaticPad``
//! to pad mel to ``staticTimeLength`` with zero rows at the tail.
enum class TimePadding
{
    kNone,
    kStaticPad, //!< Pad / truncate to ``staticTimeLength`` rows.
};

//! Configuration for one mel-spectrogram extractor instance.
struct MelExtractorConfig
{
    //! Display name used in log messages.
    std::string name;

    int32_t sampleRate{16000};
    int32_t nFFT{400};
    int32_t hopLength{160};
    int32_t winLength{400}; //!< Window length (typically == nFFT for Whisper, 400 in a 512 FFT for Parakeet).
    int32_t nMel{128};

    //! Min/max frequency for the mel filter bank. Default 0..sr/2 matches HF
    //! Whisper.
    float minFrequencyHz{0.0f};
    float maxFrequencyHz{-1.0f}; //!< Negative -> sample_rate / 2.

    //! Pre-emphasis filter ``y[t] = x[t] - coeff * x[t-1]``. Disabled when
    //! ``preemphCoeff == 0``. When ``preemphPostScale != 0`` the filtered
    //! frame is also multiplied by it.
    float preemphCoeff{0.0f};
    float preemphPostScale{0.0f};

    WindowType windowType{WindowType::kHannPeriodic};
    //! Where the window sits inside the nFFT-sized FFT input buffer when
    //! ``winLength < nFFT``. HF Whisper / Parakeet (torch.stft-style) centre
    //! the window: source ``[start+pad, start+pad+winLen)`` -> buffer
    //! ``[pad, pad+winLen)``, ``pad = (nFFT-winLen)/2``. Left-aligned mode
    //! (unfold + rfft(n=nFFT)) maps source ``[start, start+winLen)`` ->
    //! buffer ``[0, winLen)`` with trailing zeros. Ignored when winLen == nFFT.
    bool windowCentredInFft{true};
    //! When true, feed the mel filter bank the magnitude spectrum ``|stft|``
    //! (HF Gemma4) instead of the power spectrum ``|stft|^2``.
    bool magnitudeSpectrum{false};

    MelScale melScale{MelScale::kHtk};
    MelNorm melNorm{MelNorm::kSlaney};
    //! When true, build triangle filters with their slopes linear in *mel*
    //! space rather than Hz. Used together with ``MelScale::kKaldi``.
    bool triangulariseInMelSpace{false};

    LogType logType{LogType::kLog10};
    LogFloorMode logFloorMode{LogFloorMode::kMax};
    float logFloor{1e-10f}; //!< Per-FE: Whisper 1e-10, Parakeet 2^-24.

    MelLayout layout{MelLayout::kMelTime};
    PostNormalize postNormalize{PostNormalize::kWhisperClamp};
    TimePadding timePadding{TimePadding::kNone};
    int32_t staticTimeLength{0};

    FramePadding framePadding{FramePadding::kLeftAlignedZero};

    //! When true, drop the last STFT frame before mel filter / log / post-norm,
    //! matching HF Whisper / Parakeet's ``stft[..., :-1]`` and the original
    //! Whisper reference. Without this the post-normalize statistics
    //! (whisper max-clamp, parakeet per-feature mean/std) drift from HF by
    //! O(1e-1) at frame boundaries even though the underlying mel-power
    //! values are byte-identical.
    bool dropLastStftFrame{false};

    //! Pointer to a precomputed mel filter bank of shape ``[nMel × (nFFT/2 + 1)]``
    //! in row-major order. Generated offline by
    //! ``scripts/gen_mel_filter_bank.py`` and embedded as a static array.
    //! Lifetime must outlive the extractor (typically pointer to a
    //! ``static constexpr`` table).
    float const* melFilter{nullptr};
};

//! CPU mel-spectrogram extractor.
//!
//! Takes mono float32 PCM and produces a host-resident ``Tensor`` of
//! shape determined by the config's layout + padding. Pipeline mirrors
//! HF feature extractors numerically (windowing, hand-rolled radix-2
//! FFT with direct-DFT fallback for odd sizes, power spectrum, mel filter
//! mat-mul, log, optional post-normalize).
class MelExtractor
{
public:
    //! Default-constructed extractor is empty; runners move-assign one from
    //! ``makeWhisperExtractor`` / ``makeParakeetExtractor`` in
    //! ``validateAndFillConfig`` before any ``extract`` call.
    MelExtractor() noexcept;
    explicit MelExtractor(MelExtractorConfig cfg);
    ~MelExtractor();

    MelExtractor(MelExtractor const&) = delete;
    MelExtractor& operator=(MelExtractor const&) = delete;
    MelExtractor(MelExtractor&&) noexcept;
    MelExtractor& operator=(MelExtractor&&) noexcept;

    //! Extract mel-spectrogram from ``pcm`` into ``out``.
    //!
    //! ``pcm.sampleRate`` must equal ``config.sampleRate`` (caller
    //! responsibility — pass the right ``targetSampleRate`` to
    //! ``loadAudioBytes``).
    //!
    //! \param pcm Mono float32 PCM.
    //! \param out Populated with mel data on host memory (caller may
    //!            copy to device).
    //! \return true on success, false on bad input / config mismatch.
    bool extract(AudioPCM const& pcm, Tensor& out);

    MelExtractorConfig const& config() const noexcept
    {
        return mConfig;
    }

    //! Internally built mel filter bank: row-major ``[nMel x (nFFT/2 + 1)]``
    //! weights, valid for the extractor's lifetime. Empty for a
    //! default-constructed extractor or when a precomputed ``melFilter`` was
    //! supplied in the config.
    std::vector<float> const& melFilterBank() const noexcept;

    //! Analysis window used by ``extract()``: ``winLength`` float taps, valid
    //! for the extractor's lifetime. Empty for a default-constructed (empty)
    //! extractor.
    std::vector<float> const& window() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
    MelExtractorConfig mConfig;
};

//! Build a Whisper-compatible extractor (16 kHz, n_fft=400, hop=160,
//! n_mel=128, slaney mel scale + slaney norm, log10, Whisper-clamp
//! normalize, ``[n_mel, T]`` layout, HF centre-reflect framing). Used by
//! Qwen3-Omni and Qwen3-ASR.
MelExtractor makeWhisperExtractor();

//! Gemma4 / USM-style extractor matching HF Gemma4FeatureExtractor defaults.
MelExtractor makeGemma4AudioExtractor();

//! Build a Parakeet-compatible extractor (16 kHz, n_fft=512, hop=160,
//! n_mel=128, slaney scale + slaney norm, Hann non-periodic window,
//! pre-emphasis 0.97, ``log(x + 2^-24)``, per-feature mean/std normalize,
//! ``[T, n_mel]`` layout, HF centre-reflect framing). Used by
//! Nemotron-Omni.
MelExtractor makeParakeetExtractor();

//! Dispatch helper: build extractor from a string tag.
//!
//! Accepts ``"whisper"``, ``"parakeet"``. Throws
//! ``std::invalid_argument`` for unknown tags.
MelExtractor makeExtractorByName(std::string const& feType);

} // namespace audio
} // namespace rt
} // namespace trt_edgellm
