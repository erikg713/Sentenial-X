def test_predictive_model(predictive_model):
    result = predictive_model.predict("sample input")
    assert isinstance(result, dict)
    assert "score" in result
    assert "anomalies" in result
    assert 0 <= result["score"] <= 1

def test_ai_core_integration_with_simulator(predictive_model, mock_engine):
    # Simulate AI-driven decision in engine
    output = mock_engine.run()
    ai_result = predictive_model.predict(output)
    assert ai_result["score"] > 0.5
