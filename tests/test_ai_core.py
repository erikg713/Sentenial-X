from scripts import load_plugins, run_pipeline_prod, run_pipeline

def test_load_plugins_functionality():
    loaded = load_plugins.load_all_plugins()
    assert isinstance(loaded, list)

def test_run_pipeline_sequential(emulation_manager, mock_engine):
    from scripts.run_pipeline import run_pipeline
    run_pipeline(sequential=True)  # Should complete without exceptions

def test_run_pipeline_prod_sequential():
    # Production pipeline should not raise errors
    run_pipeline_prod.run_pipeline_prod(sequential=True)
