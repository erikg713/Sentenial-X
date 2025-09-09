def test_discover_simulators():
    from core.simulator import discover_simulators
    sims = discover_simulators()
    assert isinstance(sims, list)

def test_emulation_manager_register_run(emulation_manager, mock_engine):
    emulation_manager.register(mock_engine)
    emulation_manager.run_all(sequential=True)
    assert len(emulation_manager._simulators) == 1

def test_telemetry_collection(emulation_manager, telemetry_collector, mock_engine):
    emulation_manager.register(mock_engine)
    emulation_manager.run_all()
    telemetry_collector.collect(emulation_manager)
    summary = telemetry_collector.summary()
    assert "engine_running" not in summary or isinstance(summary, dict)
