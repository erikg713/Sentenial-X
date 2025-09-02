"""
ONNXWrapper
-----------
Production-ready ONNX model wrapper for Sentenial-X AI/ML modules.
Supports CPU/GPU inference, input preprocessing, and batch inference.
"""

import onnxruntime as ort
import numpy as np
import logging
from typing import Any, Dict, Optional, List

# Logger setup
logger = logging.getLogger("sentenialx.ml.onnx")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ONNXWrapper:
    def __init__(self, model_path: str, use_cuda: bool = False, providers: Optional[List[str]] = None):
        """
        Initialize ONNX runtime session.
        :param model_path: Path to the .onnx model file
        :param use_cuda: Whether to use GPU (requires ONNX GPU providers)
        :param providers: Optional custom providers list
        """
        self.model_path = model_path
        self.providers = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"])
        logger.info(f"Loading ONNX model: {model_path} with providers: {self.providers}")
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"ONNX model inputs: {self.input_names}, outputs: {self.output_names}")

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the model.
        :param inputs: Dictionary mapping input names to numpy arrays
        :return: Dictionary mapping output names to numpy arrays
        """
        for name in self.input_names:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

        # Ensure inputs are numpy arrays
        input_feed = {name: np.array(inputs[name]) for name in self.input_names}
        outputs = self.session.run(self.output_names, input_feed)
        result = {name: output for name, output in zip(self.output_names, outputs)}
        return result

    def get_input_names(self) -> List[str]:
        return self.input_names

    def get_output_names(self) -> List[str]:
        return self.output_names

    def run_batch(self, batch_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of inputs.
        :param batch_inputs: List of input dictionaries
        :return: List of output dictionaries
        """
        results = []
        for inputs in batch_inputs:
            results.append(self.predict(inputs))
        return results


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Dummy ONNX model test (replace with real path)
    dummy_model_path = "model.onnx"
    try:
        wrapper = ONNXWrapper(dummy_model_path, use_cuda=False)
        dummy_input = {wrapper.input_names[0]: np.random.randn(1, 10).astype(np.float32)}
        output = wrapper.predict(dummy_input)
        print("Inference output:", output)
    except Exception as e:
        logger.error("ONNXWrapper test failed: %s", e)
