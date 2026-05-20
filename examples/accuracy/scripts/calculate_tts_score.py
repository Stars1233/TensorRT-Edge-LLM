# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Calculate TTS quality metrics (WER + SIM) for speech outputs.

Implements the exact evaluation methodology from Seed-TTS-eval
(https://github.com/BytedanceSpeech/seed-tts-eval):

  WER - Word Error Rate
    English: Whisper-large-v3 (OpenAI)
    Chinese: Paraformer-zh (FunASR / Alibaba DAMO)

  SIM - Speaker Similarity
    WavLM-large speaker embeddings (optionally load a finetuned checkpoint
    with --wavlm_checkpoint)

Text normalization follows Seed-TTS-eval: strip punctuation, lowercase for
English, character-level spacing for Chinese, zhconv for simplified Chinese.

Usage:
    # WER only (Talker audio vs Thinker text, English)
    python calculate_tts_score.py \\
        --predictions_file output.json \\
        --audio_dir audio_output/ \\
        --language en

    # WER only (Chinese, uses Paraformer-zh)
    python calculate_tts_score.py \\
        --predictions_file output.json \\
        --audio_dir audio_output/ \\
        --language zh

    # WER + SIM (with reference audio)
    python calculate_tts_score.py \\
        --predictions_file output.json \\
        --audio_dir audio_output/ \\
        --reference_audio_dir hf_audio/ \\
        --wavlm_checkpoint /path/to/wavlm_large_finetune.pth \\
        --language en

    # WER with explicit reference text
    python calculate_tts_score.py \\
        --predictions_file output.json \\
        --audio_dir audio_output/ \\
        --references_file references.json \\
        --language zh
"""

import argparse
import json
import os
import string
from typing import Dict, Optional

try:
    from zhon.hanzi import punctuation as zh_punctuation
except ImportError:
    zh_punctuation = ""

ALL_PUNCTUATION = zh_punctuation + string.punctuation

# ---------------------------------------------------------------------------
# Text normalization (matches Seed-TTS-eval run_wer.py)
# ---------------------------------------------------------------------------


def normalize_text(text: str, language: str) -> str:
    """
    Normalize text for WER computation following Seed-TTS-eval conventions.

    Args:
        text: Raw text string.
        language: 'zh' or 'en'.

    Returns:
        Normalized text.
    """
    for ch in ALL_PUNCTUATION:
        if ch == "'":
            continue
        text = text.replace(ch, "")

    text = " ".join(text.split())

    if language == "zh":
        text = " ".join(list(text.replace(" ", "")))
    elif language == "en":
        text = text.lower()

    return text


def compute_wer_single(hypothesis: str, reference: str,
                       language: str) -> Dict[str, float]:
    """
    Compute WER for a single pair after normalization.

    Args:
        hypothesis: ASR transcription.
        reference: Ground truth text.
        language: 'zh' or 'en'.

    Returns:
        Dict with 'wer', 'substitutions', 'deletions', 'insertions'.
    """
    from jiwer import process_words

    hyp = normalize_text(hypothesis, language)
    ref = normalize_text(reference, language)

    if not ref.strip():
        return {
            "wer": 0.0,
            "substitutions": 0.0,
            "deletions": 0.0,
            "insertions": 0.0
        }

    measures = process_words(ref, hyp)
    ref_len = len(ref.split())
    return {
        "wer": measures.wer,
        "substitutions": measures.substitutions / ref_len,
        "deletions": measures.deletions / ref_len,
        "insertions": measures.insertions / ref_len,
    }


# ---------------------------------------------------------------------------
# ASR engines
# ---------------------------------------------------------------------------


def load_whisper(device: str = "cuda"):
    """Load Whisper-large-v3 for English ASR."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_id,
                                                            torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return processor, model


def load_paraformer():
    """Load Paraformer-zh for Chinese ASR."""
    from funasr import AutoModel
    model = AutoModel(model="paraformer-zh")
    return model


def transcribe_en_batch(audio_paths: list,
                        processor,
                        model,
                        device: str = "cuda",
                        batch_size: int = 16) -> list:
    """Batch transcribe English audio with Whisper-large-v3."""
    import scipy.signal
    import soundfile as sf
    import torch

    results = [""] * len(audio_paths)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english",
                                                          task="transcribe")

    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start:start + batch_size]
        wavs = []
        for p in batch_paths:
            wav, sr = sf.read(p)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            wavs.append(wav)

        inputs = processor(wavs,
                           sampling_rate=16000,
                           return_tensors="pt",
                           padding=True).input_features
        dtype = next(model.parameters()).dtype
        inputs = inputs.to(device=device, dtype=dtype)

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs, forced_decoder_ids=forced_decoder_ids)
        texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        for i, text in enumerate(texts):
            results[start + i] = text.strip()
    return results


def transcribe_zh_batch(audio_paths: list, model) -> list:
    """Batch transcribe Chinese audio with Paraformer-zh."""
    import zhconv

    results = []
    res_list = model.generate(input=audio_paths, batch_size_s=300)
    for res in res_list:
        text = res["text"] if isinstance(res, dict) else res[0]["text"]
        results.append(zhconv.convert(text, "zh-cn"))
    return results


# ---------------------------------------------------------------------------
# Speaker Similarity (WavLM-large, optional finetuned checkpoint)
# ---------------------------------------------------------------------------


def load_wavlm_speaker_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load WavLM-large for speaker similarity, optionally with a finetuned
    speaker-verification checkpoint.

    The checkpoint is from Seed-TTS-eval:
    https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP

    Uses UniSpeech speaker verification code pattern.

    Args:
        checkpoint_path: Path to wavlm_large_finetune.pth.
        device: Compute device.

    Returns:
        Loaded model on device.
    """
    import torch
    from torchaudio.pipelines import WAVLM_LARGE

    bundle = WAVLM_LARGE
    model = bundle.get_model().to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path,
                                map_location=device,
                                weights_only=True)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def _load_wav_tensor(wav_path: str, target_sr: int = 16000):
    """Load a wav file as a 1-D float32 tensor at target_sr."""
    import soundfile as sf
    import torch
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        return t.squeeze(0)
    return torch.from_numpy(audio)


def _extract_embeddings_batch(wav_paths: list,
                              model,
                              device: str = "cuda",
                              batch_size: int = 32) -> list:
    """
    Extract WavLM speaker embeddings for a list of wav files in batches.

    Args:
        wav_paths: List of wav file paths.
        model: Loaded WavLM model.
        device: Compute device.
        batch_size: Number of wavs per GPU batch.

    Returns:
        List of 1-D embedding tensors (on CPU).
    """
    import torch

    max_audio_samples = 16000 * 30  # Cap at 30 seconds to avoid OOM

    embeddings = [None] * len(wav_paths)
    loaded = []
    for i, p in enumerate(wav_paths):
        try:
            wav = _load_wav_tensor(p)
            if wav.shape[0] > max_audio_samples:
                wav = wav[:max_audio_samples]
            loaded.append((i, wav))
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
            embeddings[i] = None

    for start in range(0, len(loaded), batch_size):
        batch = loaded[start:start + batch_size]
        waveforms = [wav for _, wav in batch]
        lengths = torch.tensor([wav.shape[0] for wav in waveforms],
                               device=device)
        padded = torch.nn.utils.rnn.pad_sequence(waveforms,
                                                 batch_first=True).to(device)
        with torch.no_grad():
            features, feature_lengths = model.extract_features(padded, lengths)
            final_features = features[-1]

        if feature_lengths is None:
            batch_embeddings = final_features.mean(dim=1)
        else:
            steps = torch.arange(final_features.shape[1],
                                 device=device).unsqueeze(0)
            mask = steps < feature_lengths.unsqueeze(1)
            masked = final_features * mask.unsqueeze(-1)
            batch_embeddings = masked.sum(
                dim=1) / feature_lengths.clamp_min(1).unsqueeze(1)

        for (idx, _), embedding in zip(batch, batch_embeddings):
            embeddings[idx] = embedding.cpu()

    return embeddings


def compute_speaker_similarity_batch(
    test_paths: list,
    ref_paths: list,
    model,
    device: str = "cuda",
    batch_size: int = 32,
) -> list:
    """
    Batch compute cosine similarity of speaker embeddings using WavLM.

    Args:
        test_paths: List of test audio wav paths.
        ref_paths: List of reference audio wav paths (same length).
        model: Loaded WavLM model.
        device: Compute device.
        batch_size: Batch size for WavLM inference.

    Returns:
        List of cosine similarity scores.
    """
    import torch

    all_paths = test_paths + ref_paths
    all_embs = _extract_embeddings_batch(all_paths, model, device, batch_size)

    n = len(test_paths)
    similarities = []
    for i in range(n):
        emb_test = all_embs[i]
        emb_ref = all_embs[n + i]
        if emb_test is None or emb_ref is None:
            similarities.append(None)
        else:
            sim = torch.nn.functional.cosine_similarity(
                emb_test.unsqueeze(0), emb_ref.unsqueeze(0)).item()
            similarities.append(sim)
    return similarities


# ---------------------------------------------------------------------------
# Audio file lookup
# ---------------------------------------------------------------------------


def find_audio_file(audio_dir: str, request_idx: int,
                    batch_idx: int) -> Optional[str]:
    """
    Locate wav file for a given request/batch index.

    Args:
        audio_dir: Directory containing wav files.
        request_idx: Request index from output JSON.
        batch_idx: Batch index from output JSON.

    Returns:
        Path to wav file, or None if not found.
    """
    filename = f"audio_req{request_idx}_batch{batch_idx}.wav"
    path = os.path.join(audio_dir, filename)
    if os.path.exists(path):
        return path
    if batch_idx == 0:
        filename = f"audio_req{request_idx}.wav"
        path = os.path.join(audio_dir, filename)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main function to calculate audio quality scores."""
    parser = argparse.ArgumentParser(
        description="Calculate audio WER and speaker similarity "
        "(Seed-TTS-eval methodology)")
    parser.add_argument("--predictions_file",
                        type=str,
                        required=True,
                        help="Path to inference output JSON file")
    parser.add_argument("--audio_dir",
                        type=str,
                        required=True,
                        help="Directory containing test audio wav files")
    parser.add_argument(
        "--references_file",
        type=str,
        default=None,
        help="Optional references JSON for WER. "
        "If not provided, uses output_text from predictions_file")
    parser.add_argument(
        "--reference_audio_dir",
        type=str,
        default=None,
        help="Directory with reference audio wav files for SIM")
    parser.add_argument(
        "--wavlm_checkpoint",
        type=str,
        default=None,
        help="Optional path to a finetuned wavlm_large_finetune.pth checkpoint "
        "for speaker similarity")
    parser.add_argument("--language",
                        type=str,
                        required=True,
                        choices=["en", "zh"],
                        help="Language: 'en' (Whisper) or 'zh' (Paraformer)")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="Device for model inference (default: cuda)")
    parser.add_argument("--skip_wer",
                        action="store_true",
                        help="Skip WER computation")
    parser.add_argument("--skip_sim",
                        action="store_true",
                        help="Skip speaker similarity computation")
    parser.add_argument("--output_file",
                        type=str,
                        default=None,
                        help="Path to save detailed results JSON")

    args = parser.parse_args()

    run_wer = not args.skip_wer
    run_sim = not args.skip_sim

    # Load predictions
    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    references_data = None
    if args.references_file:
        with open(args.references_file, 'r', encoding='utf-8') as f:
            references_data = json.load(f)

    error_message = "TensorRT Edge LLM cannot handle this request. Fails."

    # Load ASR model
    whisper_processor, whisper_model, paraformer_model = None, None, None
    if run_wer:
        if args.language == "en":
            print("Loading Whisper-large-v3 for English ASR...")
            whisper_processor, whisper_model = load_whisper(args.device)
        elif args.language == "zh":
            print("Loading Paraformer-zh for Chinese ASR...")
            paraformer_model = load_paraformer()

    # Load WavLM for speaker similarity (only if reference audio is available)
    wavlm_model = None
    sim_model_name = None
    has_ref_audio = (args.reference_audio_dir is not None)
    if not has_ref_audio and references_data:
        has_ref_audio = any(
            r.get("reference_audio")
            for r in references_data.get("requests", []))
    if run_sim and has_ref_audio:
        has_finetuned_checkpoint = (args.wavlm_checkpoint is not None
                                    and os.path.exists(args.wavlm_checkpoint))
        if args.wavlm_checkpoint is None:
            print("Warning: --wavlm_checkpoint not provided, "
                  "using base WavLM-large without finetuning")
        print("Loading WavLM-large for speaker similarity...")
        wavlm_model = load_wavlm_speaker_model(args.wavlm_checkpoint,
                                               args.device)
        sim_model_name = ("WavLM-large finetuned"
                          if has_finetuned_checkpoint else "WavLM-large base")

    # Collect valid samples
    responses = predictions_data["responses"]
    if references_data:
        requests = references_data["requests"]
    else:
        requests = [None] * len(responses)

    valid_samples = []
    skipped_count = 0
    missing_audio_count = 0

    for response, request in zip(responses, requests):
        output_text = response["output_text"]
        request_idx = response.get("request_idx", 0)
        batch_idx = response.get("batch_idx", 0)

        if output_text == error_message:
            skipped_count += 1
            continue

        audio_path = find_audio_file(args.audio_dir, request_idx, batch_idx)
        if audio_path is None:
            missing_audio_count += 1
            continue

        ref_text = (request.get("reference", "")
                    if request is not None else output_text)
        ref_audio = None
        if request is not None and request.get("reference_audio"):
            p = request["reference_audio"]
            if os.path.exists(p):
                ref_audio = p
        if ref_audio is None and args.reference_audio_dir:
            ref_audio = find_audio_file(args.reference_audio_dir, request_idx,
                                        batch_idx)

        valid_samples.append({
            "request_idx": request_idx,
            "batch_idx": batch_idx,
            "audio_file": audio_path,
            "output_text": output_text,
            "ref_text": ref_text,
            "ref_audio": ref_audio,
        })

    print(f"  Valid samples: {len(valid_samples)}, "
          f"skipped: {skipped_count}, missing audio: {missing_audio_count}")

    # --- Batch ASR for WER ---
    transcriptions = [None] * len(valid_samples)
    if run_wer and valid_samples:
        audio_paths = [s["audio_file"] for s in valid_samples]
        print(f"  Batch ASR ({len(audio_paths)} files)...")
        try:
            if args.language == "en":
                transcriptions = transcribe_en_batch(audio_paths,
                                                     whisper_processor,
                                                     whisper_model,
                                                     args.device,
                                                     batch_size=16)
            else:
                transcriptions = transcribe_zh_batch(audio_paths,
                                                     paraformer_model)
            print(f"  ASR done")
        except Exception as e:
            print(f"  Batch ASR failed, falling back to sequential: {e}")
            for i, path in enumerate(audio_paths):
                try:
                    if args.language == "en":
                        transcriptions[i] = transcribe_en_batch(
                            [path],
                            whisper_processor,
                            whisper_model,
                            args.device,
                            batch_size=1)[0]
                    else:
                        transcriptions[i] = transcribe_zh_batch(
                            [path], paraformer_model)[0]
                except Exception:
                    transcriptions[i] = ""

    # --- Batch SIM ---
    sim_values = []
    sim_results = [None] * len(valid_samples)
    if run_sim and wavlm_model is not None and valid_samples:
        sim_indices = [
            i for i, s in enumerate(valid_samples) if s["ref_audio"]
        ]
        if sim_indices:
            test_paths = [valid_samples[i]["audio_file"] for i in sim_indices]
            ref_paths = [valid_samples[i]["ref_audio"] for i in sim_indices]
            print(f"  Batch SIM ({len(sim_indices)} pairs, "
                  f"batch_size=32)...")
            sims = compute_speaker_similarity_batch(test_paths,
                                                    ref_paths,
                                                    wavlm_model,
                                                    args.device,
                                                    batch_size=32)
            for j, idx in enumerate(sim_indices):
                if sims[j] is not None:
                    sim_results[idx] = sims[j]
                    sim_values.append(sims[j])
            print(f"  SIM done")

    # --- Assemble per_sample and compute WER ---
    per_sample = []
    wer_values = []
    for i, s in enumerate(valid_samples):
        sample = {
            "request_idx": s["request_idx"],
            "batch_idx": s["batch_idx"],
            "audio_file": s["audio_file"],
            "output_text": s["output_text"],
        }

        if run_wer and transcriptions[i] is not None:
            hyp = transcriptions[i]
            metrics = compute_wer_single(hyp, s["ref_text"], args.language)
            wer_values.append(metrics["wer"])
            sample["reference"] = s["ref_text"]
            sample["transcription"] = hyp
            sample["wer"] = round(metrics["wer"], 6)

        if sim_results[i] is not None:
            sample["speaker_similarity"] = round(sim_results[i], 6)
            sample["reference_audio"] = s["ref_audio"]

        per_sample.append(sample)

    # --- Report ---
    total_count = len(responses)
    if skipped_count > 0:
        print(
            f"Skipped {skipped_count}/{total_count} entries with error messages"
        )
    if missing_audio_count > 0:
        print(f"Missing audio for {missing_audio_count}/{total_count} entries")

    valid_count = len(per_sample)
    if valid_count == 0:
        print("No valid audio samples to evaluate")
        return {}

    result = {
        "language": args.language,
        "total_count": total_count,
        "valid_count": valid_count,
        "skipped_count": skipped_count,
        "missing_audio_count": missing_audio_count,
    }

    print(f"\nAudio Score Results (evaluated {valid_count} samples):")
    print(f"  Language:  {args.language}")
    if args.language == "en":
        print(f"  ASR:       Whisper-large-v3")
    else:
        print(f"  ASR:       Paraformer-zh")

    if wer_values:
        mean_wer = sum(wer_values) / len(wer_values)
        result["wer"] = round(mean_wer, 6)
        catastrophic = [w for w in wer_values if w >= 0.8]
        result["catastrophic_failure_count"] = len(catastrophic)
        result["catastrophic_failure_rate"] = round(
            len(catastrophic) / len(wer_values), 4) if wer_values else 0.0
        clean_wer_values = [w for w in wer_values if w < 0.8]
        result["wer_excluding_failures"] = round(
            sum(clean_wer_values) /
            len(clean_wer_values), 6) if clean_wer_values else 0.0
        print(f"  WER:       {mean_wer:.4f} ({mean_wer*100:.2f}%)")
        print(f"  WER (excl catastrophic): "
              f"{result['wer_excluding_failures']:.4f} "
              f"({result['wer_excluding_failures']*100:.2f}%)")
        print(f"  Catastrophic failures (WER>=0.8): "
              f"{len(catastrophic)}/{len(wer_values)} "
              f"({result['catastrophic_failure_rate']*100:.1f}%)")

    if sim_values:
        mean_sim = sum(sim_values) / len(sim_values)
        result["speaker_similarity"] = round(mean_sim, 6)
        result["speaker_similarity_model"] = sim_model_name or "WavLM-large"
        print(f"  SIM:       {mean_sim:.4f}")
        print(f"  SIM model: {result['speaker_similarity_model']}")

    result["per_sample"] = per_sample

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {args.output_file}")

    return result


if __name__ == "__main__":
    main()
