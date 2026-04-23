"""
Synthesizer factory.

Usage:
    from novel_view_synthesis import get_synthesizer
    synth = get_synthesizer("svd")          # lazy – model not loaded yet
    synth.load_model("cuda")                # load weights
    paths = synth.generate_views(img, out)  # run inference
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseSynthesizer

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "svd": "novel_view_synthesis.svd_synthesizer.SVDSynthesizer",
    "viewcrafter": "novel_view_synthesis.viewcrafter_synthesizer.ViewCrafterSynthesizer",
    "vivid": "novel_view_synthesis.vivid_synthesizer.VIVIDSynthesizer",
    "panodreamer": "novel_view_synthesis.panodreamer_synthesizer.PanoDreamerSynthesizer",
    "zero123pp": "novel_view_synthesis.zero123pp_synthesizer.Zero123PPSynthesizer",
    "sv3d": "novel_view_synthesis.sv3d_synthesizer.SV3DSynthesizer",
    "seva": "novel_view_synthesis.seva_synthesizer.SevaSynthesizer",
    "seva_4070ti": "novel_view_synthesis.seva_synthesizer_4070ti.SevaSynthesizer",
}

AVAILABLE_MODELS = list(_REGISTRY.keys())


def get_synthesizer(name: str) -> "BaseSynthesizer":
    """Instantiate a synthesizer by short name (e.g. ``"svd"``)."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown synthesizer '{name}'. Choose from {AVAILABLE_MODELS}"
        )

    module_path, class_name = _REGISTRY[name].rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls()
    logger.info(f"Instantiated synthesizer: {cls.__name__}")
    return instance
