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
"""Calibration datasets for quantization, selected by name.

A calibration dataset is just a **generator function** registered under a
name, per modality. The quantization CLI passes only the name
(``--text_dataset`` / ``--image_dataset`` / ``--audio_dataset``); an unknown
name fails out with the list of registered datasets and a pointer to the
customization guide and template, instead of silently falling back.

The three modality **contracts** are the functions' yield types -- no base
classes, no instances:

    text  : () -> Iterator[str]                      # one document per item
    image : () -> Iterator[tuple[image, question]]   # image: PIL / path / URL
    audio : () -> Iterator[tuple[audio_bytes, str]]  # encoded audio + transcript

To add a custom dataset, copy ``custom_dataset_template.py``, write a
generator that yields the right type, decorate it with ``@register(...)``, and
import it so registration runs. See
``docs/source/developer_guide/customization/calibration-datasets.md``.
"""

import os
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

# Modality identifiers shared by the registry, the contracts, and the CLI.
TEXT = "text"
IMAGE = "image"
AUDIO = "audio"
MODALITIES = (TEXT, IMAGE, AUDIO)

# The contracts, as type aliases. A registered dataset is a zero-argument
# generator function producing samples of the modality's type.
TextDataset = Callable[[], Iterator[str]]
ImageDataset = Callable[[], Iterator[Tuple[Any, str]]]
AudioDataset = Callable[[], Iterator[Tuple[bytes, str]]]
CalibrationDataset = Callable[[], Iterator[Any]]

# Default dataset name per modality, used when the CLI flag is left unset.
DEFAULT_TEXT_DATASET = "cnn_dailymail"
DEFAULT_IMAGE_DATASET = "mmmu"
DEFAULT_AUDIO_DATASET = "librispeech"
_DEFAULTS = {
    TEXT: DEFAULT_TEXT_DATASET,
    IMAGE: DEFAULT_IMAGE_DATASET,
    AUDIO: DEFAULT_AUDIO_DATASET,
}

# Where the unknown-dataset error sends users to add their own dataset.
CUSTOMIZATION_GUIDE = (
    "docs/source/developer_guide/customization/calibration-datasets.md")
TEMPLATE_FILE = (
    "tensorrt_edgellm/quantization/datasets/custom_dataset_template.py")

# modality -> {name: generator function}
_REGISTRY = {TEXT: {}, IMAGE: {}, AUDIO: {}}


class UnknownCalibrationDataset(KeyError):
    """Raised when a requested calibration dataset name is not registered."""

    def __init__(self, modality: str, name: str, available: List[str]):
        self.modality = modality
        self.dataset_name = name
        self.available = available
        listed = ", ".join(available) if available else "(none registered)"
        super().__init__(
            f"Unknown {modality} calibration dataset {name!r}. "
            f"Registered {modality} datasets: {listed}. To calibrate on a "
            f"custom dataset, write a {modality} generator and register it "
            f"with @register({modality!r}, ...). See the customization guide "
            f"({CUSTOMIZATION_GUIDE}) and copy the template at "
            f"{TEMPLATE_FILE}.")

    def __str__(self) -> str:
        return self.args[0]  # KeyError.__str__ would re-quote the message


def register(modality: str, name: str):
    """Decorator: register a generator function as a calibration dataset.

    ``modality`` is ``"text"`` / ``"image"`` / ``"audio"``; ``name`` is the
    value users pass on the CLI. Re-registering a name replaces the previous
    entry, so a custom dataset can override a built-in by reusing its name.
    """
    if modality not in _REGISTRY:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {MODALITIES}.")
    if not name:
        raise ValueError("Calibration dataset name must be non-empty.")

    def _decorator(fn: CalibrationDataset) -> CalibrationDataset:
        fn.calib_modality = modality
        fn.calib_name = name
        _REGISTRY[modality][name] = fn
        return fn

    return _decorator


def available_datasets(modality: str) -> List[str]:
    """Return the sorted registered dataset names for *modality*."""
    if modality not in _REGISTRY:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {MODALITIES}.")
    return sorted(_REGISTRY[modality])


def get_dataset(modality: str, name: str) -> CalibrationDataset:
    """Return the generator function registered under ``name`` for *modality*.

    Raises :class:`UnknownCalibrationDataset` (pointing at the customization
    guide) when no dataset is registered under that name.
    """
    if modality not in _REGISTRY:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {MODALITIES}.")
    fn = _REGISTRY[modality].get(name)
    if fn is None:
        raise UnknownCalibrationDataset(modality, name,
                                        available_datasets(modality))
    return fn


def resolve_dataset(
    value: Union[str, CalibrationDataset, None],
    modality: str,
) -> CalibrationDataset:
    """Resolve a name / generator / ``None`` to a dataset generator function.

    * ``None``     -> the modality's default registered dataset.
    * ``str``      -> looked up in the registry (fails out if unknown).
    * a callable   -> used directly (lets the Python API pass a one-off
      generator without registering it).
    """
    if value is None:
        return get_dataset(modality, _DEFAULTS[modality])
    if isinstance(value, str):
        return get_dataset(modality, value)
    if callable(value):
        return value
    raise TypeError(
        f"{modality} dataset must be a registered name (str) or a generator "
        f"function; got {type(value).__name__}.")


def dataset_name(fn: CalibrationDataset) -> str:
    """Return the registered name of a dataset function for logging."""
    return getattr(fn, "calib_name", getattr(fn, "__name__", "<dataset>"))


# --- Shared loading helpers (used by built-ins and custom datasets) --------

_OVERRIDE_PREFIX = "EDGELLM_QUANT_DATASET_"


def local_override_path(name: str) -> Optional[str]:
    """Return a local dataset path from this dataset's env override, if set.

    Built-in datasets load from the Hugging Face Hub by default. CI and
    air-gapped hosts can point a built-in at a cached copy by setting
    ``EDGELLM_QUANT_DATASET_<NAME>`` (NAME upper-cased, ``-`` / ``/`` -> ``_``)
    to a local JSON/JSONL file, a ``load_dataset``-style dataset directory, or
    a ``datasets.save_to_disk`` directory. The CLI still only passes the name.
    """
    env = _OVERRIDE_PREFIX + name.upper().replace("-", "_").replace("/", "_")
    return os.environ.get(env) or None


def first_present(example: dict, fields: Tuple[str, ...]) -> Any:
    """Return the first non-``None`` value among *fields* in *example*."""
    for field in fields:
        value = example.get(field)
        if value is not None:
            return value
    return None


def load_hf_split(hf_id: str,
                  *,
                  name: str,
                  split: str,
                  config: Optional[str] = None,
                  streaming: bool = False):
    """Load one dataset split for built-in *name* (honouring the env override).

    With ``EDGELLM_QUANT_DATASET_<NAME>`` set, loads from that local path (a
    JSON/JSONL file, a ``save_to_disk`` directory, or a ``load_dataset``-style
    directory); otherwise from the Hub id.
    """
    from datasets import DatasetDict, load_dataset, load_from_disk

    local_path = local_override_path(name)
    if local_path:
        if os.path.isfile(local_path):
            return load_dataset("json",
                                data_files={split: local_path},
                                split=split)
        if os.path.isdir(local_path):
            # ``save_to_disk`` output has these sentinel files; anything else
            # is treated as a ``load_dataset``-style dataset directory.
            is_saved = (
                os.path.exists(os.path.join(local_path, "dataset_dict.json"))
                or os.path.exists(os.path.join(local_path, "state.json")))
            if is_saved:
                ds = load_from_disk(local_path)
                if isinstance(ds, DatasetDict):
                    if split not in ds:
                        raise ValueError(
                            f"Local dataset {local_path!r} has no split "
                            f"{split!r}; found {list(ds.keys())}.")
                    return ds[split]
                return ds
            if config:
                return load_dataset(local_path, config, split=split)
            return load_dataset(local_path, split=split)
        raise FileNotFoundError(
            f"Local dataset override {local_path!r} is neither a file nor a "
            f"directory.")

    kwargs = {"split": split}
    if streaming:
        kwargs["streaming"] = True
    if config:
        return load_dataset(hf_id, config, **kwargs)
    return load_dataset(hf_id, **kwargs)


# Import the built-in datasets so their @register decorators run on import.
from . import builtin as _builtin  # noqa: E402,F401  isort:skip

__all__ = [
    "TEXT",
    "IMAGE",
    "AUDIO",
    "MODALITIES",
    "TextDataset",
    "ImageDataset",
    "AudioDataset",
    "CalibrationDataset",
    "register",
    "get_dataset",
    "resolve_dataset",
    "available_datasets",
    "dataset_name",
    "UnknownCalibrationDataset",
    "load_hf_split",
    "first_present",
    "local_override_path",
    "DEFAULT_TEXT_DATASET",
    "DEFAULT_IMAGE_DATASET",
    "DEFAULT_AUDIO_DATASET",
    "CUSTOMIZATION_GUIDE",
    "TEMPLATE_FILE",
]
