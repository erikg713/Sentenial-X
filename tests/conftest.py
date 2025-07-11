import pytest

@pytest.fixture
def mock_request():
    return {
        "headers": {},
        "body": "",
        "query": ""
    }

