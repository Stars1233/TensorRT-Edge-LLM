# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Nemotron-3.5-ASR RNN-T decoder step: prediction network + joint, fused.

One engine invocation = one greedy-decode step:

    (decoder_input_ids, hidden_state, cell_state, encoder_frame)
        → (logits, present_hidden_state, present_cell_state)

The prediction network is a 2-layer LSTM over emitted non-blank tokens; the
joint is ``head(relu(encoder_frame + decoder_out))``. The LSTM is unrolled
manually (single time step) so the ONNX graph is plain MatMul/activation ops
rather than ``aten::lstm``.

RNN-T semantics live in the RUNTIME loop, not this graph:
  - blank prediction → advance the encoder frame pointer and DISCARD
    ``present_{hidden,cell}_state`` (keep feeding the old state — this
    reproduces HF's "blank does not update the LSTM" rule and its
    all-blank fast path in one mechanism);
  - non-blank → emit the token, ADOPT the present states, keep the frame;
  - force an advance after ``max_symbols_per_step`` (10) non-blanks;
  - stop when the frame pointer passes the last encoder frame.

State is engine I/O (input state → present state), following the
Nemotron-H recurrent-state pattern.

Checkpoint weight key prefixes (identity mapping):
    ``decoder.embedding.weight``      [vocab, H]
    ``decoder.lstm.{weight,bias}_{ih,hh}_l{0,1}``
    ``decoder.decoder_projector.{weight,bias}``
    ``joint.head.{weight,bias}``
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Manually-stepped multi-layer LSTM (parameter names match ``nn.LSTM``)
# ---------------------------------------------------------------------------


class _LSTMStep(nn.Module):
    """Single-time-step multi-layer LSTM with ``nn.LSTM`` parameter naming.

    Gate order per PyTorch convention: input, forget, cell, output.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            setattr(self, f"weight_ih_l{layer}",
                    nn.Parameter(torch.zeros(4 * hidden_size, in_size)))
            setattr(self, f"weight_hh_l{layer}",
                    nn.Parameter(torch.zeros(4 * hidden_size, hidden_size)))
            setattr(self, f"bias_ih_l{layer}",
                    nn.Parameter(torch.zeros(4 * hidden_size)))
            setattr(self, f"bias_hh_l{layer}",
                    nn.Parameter(torch.zeros(4 * hidden_size)))

    def forward(
        self, x: torch.Tensor, hidden_state: torch.Tensor,
        cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_size]
            hidden_state / cell_state: [num_layers, B, hidden_size]

        Returns:
            (output [B, hidden_size], present_hidden, present_cell)
        """
        new_h, new_c = [], []
        for layer in range(self.num_layers):
            w_ih = getattr(self, f"weight_ih_l{layer}")
            w_hh = getattr(self, f"weight_hh_l{layer}")
            b_ih = getattr(self, f"bias_ih_l{layer}")
            b_hh = getattr(self, f"bias_hh_l{layer}")
            h_prev = hidden_state[layer]
            c_prev = cell_state[layer]

            gates = (F.linear(x, w_ih, b_ih) + F.linear(h_prev, w_hh, b_hh))
            i, f, g, o = gates.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            new_h.append(h)
            new_c.append(c)
            x = h
        return x, torch.stack(new_h, dim=0), torch.stack(new_c, dim=0)


# ---------------------------------------------------------------------------
# Prediction network and joint (attribute names match checkpoint keys)
# ---------------------------------------------------------------------------


class _RNNTPredictionNetwork(nn.Module):
    """Checkpoint keys under ``decoder.``: embedding, lstm, decoder_projector."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = _LSTMStep(hidden_size, hidden_size, num_layers)
        self.decoder_projector = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, input_ids: torch.Tensor, hidden_state: torch.Tensor,
        cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids).squeeze(1)  # [B, 1] → [B, H]
        x, present_h, present_c = self.lstm(x, hidden_state, cell_state)
        return self.decoder_projector(x), present_h, present_c


class _RNNTJoint(nn.Module):
    """Checkpoint keys under ``joint.``: head."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_frame: torch.Tensor,
                decoder_out: torch.Tensor) -> torch.Tensor:
        return self.head(F.relu(encoder_frame + decoder_out))


# ---------------------------------------------------------------------------
# Fused step model
# ---------------------------------------------------------------------------


class Nemotron3_5AsrRNNTStepModel(nn.Module):
    """One RNN-T greedy-decode step: LSTM prediction net + joint network."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["decoder_hidden_size"]
        self.num_layers = config.get("num_decoder_layers", 2)
        self.blank_token_id = config["blank_token_id"]
        self.max_symbols_per_step = config.get("max_symbols_per_step", 10)

        self.decoder = _RNNTPredictionNetwork(self.vocab_size,
                                              self.hidden_size,
                                              self.num_layers)
        self.joint = _RNNTJoint(self.hidden_size, self.vocab_size)

    def forward(
        self, decoder_input_ids: torch.Tensor, hidden_state: torch.Tensor,
        cell_state: torch.Tensor, encoder_frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_input_ids: [1, 1] int64 — last emitted token (blank at
                start of decode).
            hidden_state / cell_state: [num_layers, 1, H] LSTM state.
            encoder_frame: [1, H] current encoder frame (joint space).

        Returns:
            logits [1, vocab_size], present_hidden_state, present_cell_state.
        """
        decoder_out, present_h, present_c = self.decoder(
            decoder_input_ids, hidden_state, cell_state)
        logits = self.joint(encoder_frame, decoder_out)
        return logits, present_h, present_c

    def get_onnx_export_args(self, config: dict, device: str):
        """Return (args, input_names, output_names, dynamic_shapes)."""
        dtype = next(self.parameters()).dtype
        args = (
            torch.full((1, 1),
                       self.blank_token_id,
                       dtype=torch.int64,
                       device=device),
            torch.zeros(self.num_layers,
                        1,
                        self.hidden_size,
                        dtype=dtype,
                        device=device),
            torch.zeros(self.num_layers,
                        1,
                        self.hidden_size,
                        dtype=dtype,
                        device=device),
            torch.zeros(1, self.hidden_size, dtype=dtype, device=device),
        )
        input_names = [
            "decoder_input_ids", "hidden_state", "cell_state", "encoder_frame"
        ]
        output_names = ["logits", "present_hidden_state", "present_cell_state"]
        dynamic_shapes = {name: None for name in input_names}
        return args, input_names, output_names, dynamic_shapes


# ---------------------------------------------------------------------------
# Weight loading / factory
# ---------------------------------------------------------------------------

_DECODER_PREFIXES = ("decoder.", "joint.")


def _load_weights(model: Nemotron3_5AsrRNNTStepModel, weights: dict) -> None:
    from ...checkpoint.loader import load_submodule_weights

    def _remap(k: str) -> "str | None":
        if k.startswith(_DECODER_PREFIXES):
            return k
        return None

    load_submodule_weights(model,
                           weights,
                           _remap,
                           label="Nemotron3_5AsrRNNTStepModel")


def build_nemotron3_5_asr_decoder(
        config: dict,
        weights: dict,
        dtype: torch.dtype = torch.float16) -> Nemotron3_5AsrRNNTStepModel:
    """Build a :class:`Nemotron3_5AsrRNNTStepModel` with loaded weights."""
    model = Nemotron3_5AsrRNNTStepModel(config)
    _load_weights(model, weights)
    # Cast AFTER loading: the checkpoint is fp32 and ``_set_tensor``
    # deliberately preserves the source dtype.
    model.to(dtype)
    model.eval()
    return model


__all__ = [
    "Nemotron3_5AsrRNNTStepModel",
    "build_nemotron3_5_asr_decoder",
]
