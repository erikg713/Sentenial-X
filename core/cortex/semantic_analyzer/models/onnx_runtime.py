# core/semantic_analyzer/models/onnx_runtime.py

import os
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logging.warning("onnxruntime not installed. Install with `pip install onnxruntime`.")

logger = logging.getLogger(__name__)


class ONNXModel:
    """
    Wrapper for ONNX Runtime inference with support for CPU/GPU execution.

    Example:
        model = ONNXModel("models/semantic.onnx", use_gpu=True)
        outputs = model.run({"input": np.array([...], dtype=np.float32)})
    """

    def __init__(self, model_path: str, use_gpu: bool = False):
        if ort is None:
            raise ImportError("onnxruntime is required for ONNXModel")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.providers = self._select_providers(use_gpu)
        self.session: Optional[ort.InferenceSession] = None
        self._load_model()

    def _select_providers(self, use_gpu: bool) -> List[str]:
        """Select execution providers (CUDA if available, else CPU)."""
        available_providers = ort.get_available_providers()
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            logger.info("Using CUDAExecutionProvider for ONNX runtime.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CPUExecutionProvider for ONNX runtime.")
        return ["CPUExecutionProvider"]

    def _load_model(self):
        """Load the ONNX model into an inference session."""
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=self.providers,
            )
            logger.info(f"Loaded ONNX model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def get_input_names(self) -> List[str]:
        """Return input tensor names."""
        return [i.name for i in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        """Return output tensor names."""
        return [o.name for o in self.session.get_outputs()]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Perform inference.

        Args:
            inputs (dict): Mapping of input_name -> numpy array

        Returns:
            dict: Mapping of output_name -> numpy array
        """
        if self.session is None:
            raise RuntimeError("ONNX session not initialized")

        try:
            outputs = self.session.run(
                None,  # all outputs
                inputs,
            )
            return {
                name: outputs[i]
                for i, name in enumerate(self.get_output_names())
            }
        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            raise


# Utility for quick loading & inference
def load_and_run(model_path: str, input_data: Dict[str, Any], use_gpu: bool = False) -> Dict[str, np.ndarray]:
    """
    Quick utility to load a model and run inference.

    Args:
        model_path (str): Path to the ONNX model
        input_data (dict): Inputs for the model
        use_gpu (bool): Whether to try CUDA

    Returns:
        dict: Outputs from inference
    """
    model = ONNXModel(model_path, use_gpu)
    return model.run(input_data)
