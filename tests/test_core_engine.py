def test_base_engine_run(mock_engine):
    result = mock_engine.run()
    assert result == "engine_running"

def test_engine_context_manager(mock_engine):
    with mock_engine as e:
        assert e is mock_engine

def test_engine_start_stop_logging(mock_engine, caplog):
    caplog.set_level("INFO")
    mock_engine.start()
    mock_engine.stop()
    assert "start" in caplog.text.lower()
    assert "stop" in caplog.text.lower()
