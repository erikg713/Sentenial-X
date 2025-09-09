import pytest
from unittest.mock import MagicMock
from core.simulator import EmulationManager, TelemetryCollector
from core.engine import BaseEngine
from ai_core.predictive_model import PredictiveModel

# ----------------------------
# Engine Fixtures
# ----------------------------
@pytest.fixture
def mock_engine():
    class DummyEngine(BaseEngine):
        def run(self, *args, **kwargs):
            return "engine_running"
    return DummyEngine()

# ----------------------------
# Simulator Fixtures
# ----------------------------
@pytest.fixture
def emulation_manager():
    return EmulationManager()

@pytest.fixture
def telemetry_collector():
    return TelemetryCollector()

# ----------------------------
# AI Core Fixtures
# ----------------------------
@pytest.fixture
def predictive_model():
    model = PredictiveModel()
    # Mock predict method
    model.predict = MagicMock(return_value={"score": 0.95, "anomalies": 1})
    return model

# ----------------------------
# Plugin Fixture
# ----------------------------
@pytest.fixture
def loaded_plugins():
    from scripts.load_plugins import load_all_plugins
    return load_all_plugins()
