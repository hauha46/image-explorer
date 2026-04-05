"""
Abstract base class for novel view synthesis backends.

All NVS models (SVD, ViewCrafter, VIVID) implement this interface so they
can be swapped in SceneProcessor without changing any pipeline logic.
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseSynthesizer(ABC):
    """Shared contract for every novel-view-synthesis backend."""

    name: str = "base"

    @abstractmethod
    def load_model(self, device: str = "cuda") -> None:
        """Load weights onto *device*.  Called once at startup or on first use."""
        ...

    @abstractmethod
    def generate_views(
        self,
        image_path: str,
        output_dir: str,
        num_views: int = 8,
        depth_map: Optional[np.ndarray] = None,
        fov_deg: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> list[str]:
        """
        Synthesize *num_views* novel views of the scene in *image_path*.

        Parameters
        ----------
        image_path : str
            Path to the source image.
        output_dir : str
            Directory where generated view PNGs are saved.
        num_views : int
            How many output frames to produce.
        depth_map : np.ndarray | None
            Optional metric / normalised depth from DepthPro.
        fov_deg : float | None
            Estimated horizontal field-of-view (degrees) from DepthPro.
        prompt : str | None
            Optional text prompt for models that support text conditioning
            (e.g. ViewCrafter).  Ignored by models without text input.

        Returns
        -------
        list[str]
            Absolute paths to the saved view images (sorted).
        """
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Release GPU memory so another model can be loaded."""
        ...
