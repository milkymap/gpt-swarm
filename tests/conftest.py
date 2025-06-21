"""
Pytest configuration and fixtures for gpt-swarm tests.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import List

from dataschema import Message, Role
from libraries.swarming import GPTSwarm


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(role=Role.USER, content="Test message content")


@pytest.fixture
def sample_messages() -> List[Message]:
    """Create a list of sample messages for testing."""
    return [
        Message(role=Role.USER, content="Hello, how are you?"),
        Message(role=Role.ASSISTANT, content="I'm doing well, thank you!"),
        Message(role=Role.USER, content="Can you help me with something?")
    ]


@pytest.fixture
def sample_conversations(sample_messages) -> List[List[Message]]:
    """Create sample conversations for testing."""
    return [
        [Message(role=Role.USER, content="Tell me about AI")],
        [Message(role=Role.USER, content="Explain quantum physics")],
        [Message(role=Role.USER, content="What is machine learning?")]
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.usage = Mock()
    mock_response.usage.total_tokens = 150
    mock_response.output_text = "This is a test response from the AI."
    
    mock_client.responses.create.return_value = mock_response
    return mock_client


@pytest.fixture
def gpt_swarm_config():
    """Configuration for GPTSwarm instance."""
    return {
        "openai_api_key": "test-api-key-123",
        "nb_tokens_per_mn": 180_000,
        "nb_requests_per_mn": 3000,
        "model_token_size": 8192
    }


@pytest.fixture
async def gpt_swarm_instance(gpt_swarm_config, mock_openai_client):
    """Create a GPTSwarm instance for testing."""
    swarm = GPTSwarm(**gpt_swarm_config)
    # Replace the actual OpenAI client with our mock
    swarm.openai_client = mock_openai_client
    return swarm


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()