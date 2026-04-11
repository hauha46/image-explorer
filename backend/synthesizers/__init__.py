"""
Synthesizer factory.

Usage:
    from synthesizers import get_synthesizer
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
    "svd": "synthesizers.svd_synthesizer.SVDSynthesizer",
    "viewcrafter": "synthesizers.viewcrafter_synthesizer.ViewCrafterSynthesizer",
    "vivid": "synthesizers.vivid_synthesizer.VIVIDSynthesizer",
    "panodreamer": "synthesizers.panodreamer_synthesizer.PanoDreamerSynthesizer",
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
