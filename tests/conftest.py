import pytest
from httpx import AsyncClient
from apps.api import app  # adjust path to your FastAPI/Flask app
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    """
    Provides a session-wide asyncio event loop for async tests.
    """
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def async_client():
    """
    Async HTTP client fixture for testing FastAPI endpoints.
    """
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

@pytest.fixture(scope="module")
def test_target_ip():
    """
    Example fixture providing a dummy target IP for exploit testing.
    """
    return "192.168.1.100"

@pytest.fixture(scope="module")
def test_exploit_name():
    """
    Example fixture providing a dummy exploit module name.
    """
    return "ms17_010_eternalblue"

# Optional: If using a test database
# @pytest.fixture(scope="module")
# async def test_db():
#     # Initialize test database connection
#     db = await setup_test_db()
#     yield db
#     await teardown_test_db(db)
@pytest.fixture
def mock_request():
    return {
        "headers": {},
        "body": "",
        "query": ""
    }

