# tests/test_encoder.py
import pytest
import torch
from libs.ml.pytorch.model import SentenialXNet
from libs.ml.pytorch.dataset import ThreatSampleDataset

@pytest.fixture
def sample_data():
    """
    Provides sample input data and labels for testing the encoder/model.
    """
    X = [[0.1 * i for i in range(128)] for _ in range(10)]  # 10 samples, 128 features
    y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]                     # Binary labels
    return X, y

@pytest.fixture
def dataset(sample_data):
    X, y = sample_data
    return ThreatSampleDataset(X, y)

@pytest.fixture
def model():
    return SentenialXNet(input_dim=128, hidden_dim=64, output_dim=2, dropout=0.3)

def test_forward_pass(model, dataset):
    """
    Ensure model forward pass produces correct output shape.
    """
    sample, label = dataset[0]
    sample = sample.unsqueeze(0)  # Add batch dimension
    output = model(sample)
    assert output.shape == (1, 2), f"Expected output shape (1,2), got {output.shape}"

def test_training_step(model, dataset):
    """
    Basic single-step training test.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    sample, label = dataset[0]
    sample = sample.unsqueeze(0)
    label = torch.tensor([label])

    model.train()
    optimizer.zero_grad()
    output = model(sample)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    # Ensure loss is a scalar and not NaN
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

def test_dataset_length(dataset):
    """
    Validate that dataset length matches number of samples.
    """
    assert len(dataset) == 10, f"Expected dataset length 10, got {len(dataset)}"
