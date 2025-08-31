# core/semantic_analyzer/models/tests/test_classifier.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.semantic_analyzer.models.classifier import ClassifierModel
from core.semantic_analyzer.models.registry import ModelRegistry


def test_classifier_predict_mock():
    with patch("core.semantic_analyzer.models.classifier.AutoModel") as mock_model, \
         patch("core.semantic_analyzer.models.classifier.AutoTokenizer") as mock_tokenizer:

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}
        mock_tokenizer.from_pretrained.return_value = tokenizer

        transformer = MagicMock()
        transformer.config.hidden_size = 8
        transformer.return_value.last_hidden_state = MagicMock()
        transformer.return_value.last_hidden_state[:, 0, :].cpu.return_value.numpy.return_value = np.random.rand(2, 8)
        mock_model.from_pretrained.return_value = transformer

        clf = ClassifierModel(model_name="mock_clf", num_labels=2)
        clf.transformer = transformer
        clf.tokenizer = tokenizer

        results = clf.predict(["test", "input"])
        assert isinstance(results, list)
        assert "label" in results[0]
        assert "score" in results[0]
