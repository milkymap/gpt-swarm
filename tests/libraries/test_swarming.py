"""
Unit tests for libraries.swarming module.
"""
import pytest
import asyncio
import zmq
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from openai import AsyncOpenAI

from libraries.swarming import GPTSwarm
from dataschema import Message, Role


class TestGPTSwarmInit:
    """Test GPTSwarm initialization."""
    
    def test_init_with_valid_parameters(self):
        """Test GPTSwarm initialization with valid parameters."""
        swarm = GPTSwarm(
            openai_api_key="test-key",
            nb_tokens_per_mn=180_000,
            nb_requests_per_mn=3000,
            model_token_size=8192
        )
        
        assert swarm.openai_api_key == "test-key"
        assert swarm.nb_tokens_per_mn == 180_000
        assert swarm.nb_requests_per_mn == 3000
        assert swarm.model_token_size == 8192
        assert swarm.period == 1 / (3000 / 60)  # 0.02
        assert swarm.collector_address == 'inproc://collector_endpoint'
        assert isinstance(swarm.openai_client, AsyncOpenAI)
    
    def test_period_calculation(self):
        """Test period calculation for different request rates."""
        # Test with 3000 requests per minute
        swarm1 = GPTSwarm("key", 180_000, 3000, 8192)
        expected_period1 = 1 / (3000 / 60)  # 0.02 seconds
        assert swarm1.period == expected_period1
        
        # Test with 1000 requests per minute
        swarm2 = GPTSwarm("key", 180_000, 1000, 8192)
        expected_period2 = 1 / (1000 / 60)  # 0.06 seconds
        assert swarm2.period == expected_period2


class TestGPTSwarmContextManager:
    """Test GPTSwarm async context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_aenter_initialization(self):
        """Test __aenter__ method initializes required attributes."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        async with swarm as initialized_swarm:
            assert hasattr(initialized_swarm, 'mutex')
            assert hasattr(initialized_swarm, 'tokens_status')
            assert hasattr(initialized_swarm, 'start_timer')
            assert hasattr(initialized_swarm, 'ctx')
            assert hasattr(initialized_swarm, 'loop')
            
            assert isinstance(initialized_swarm.mutex, asyncio.Lock)
            assert isinstance(initialized_swarm.tokens_status, asyncio.Event)
            assert isinstance(initialized_swarm.start_timer, asyncio.Event)
    
    @pytest.mark.asyncio
    async def test_aexit_cleanup(self):
        """Test __aexit__ method cleans up resources."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        # Mock the ZMQ context
        with patch('zmq.asyncio.Context') as MockContext:
            mock_ctx = Mock()
            mock_ctx.term = Mock()
            MockContext.return_value = mock_ctx
            
            # Mock signal handler to avoid issues
            with patch.object(swarm, 'loop', Mock()) as mock_loop:
                mock_loop.add_signal_handler = Mock()
                mock_loop.is_running.return_value = False
                
                async with swarm:
                    pass  # Context manager should handle cleanup
                
                # Context cleanup should be called
                mock_ctx.term.assert_called_once()


class TestGPTSwarmWorkerStrategy:
    """Test GPTSwarm worker_strategy method."""
    
    @pytest.mark.asyncio
    async def test_worker_strategy_successful_response(self):
        """Test worker_strategy with successful response."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        # Initialize required attributes
        swarm.mutex = asyncio.Lock()
        swarm.total_tokens = 0
        swarm.consumed_tokens = 0
        swarm.model_token_size = 8192
        
        # Mock response
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 150
        mock_response.output_text = "Test response"
        
        # Mock push socket
        mock_push_socket = AsyncMock()
        
        result_message, keep_loop = await swarm.worker_strategy(
            mock_push_socket, mock_response
        )
        
        assert result_message is not None
        assert result_message.role == Role.ASSISTANT
        assert result_message.content == "Test response"
        assert keep_loop is False
        assert swarm.total_tokens == 150
        assert swarm.consumed_tokens == 150
        mock_push_socket.send.assert_called_once_with(b'...')
    
    @pytest.mark.asyncio
    async def test_worker_strategy_with_error_401(self):
        """Test worker_strategy with 401 authorization error."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        swarm.mutex = asyncio.Lock()
        
        mock_push_socket = AsyncMock()
        error = Exception("401 Unauthorized")
        
        result_message, keep_loop = await swarm.worker_strategy(
            mock_push_socket, None, error
        )
        
        assert result_message is None
        assert keep_loop is False
    
    @pytest.mark.asyncio
    async def test_worker_strategy_with_rate_limit_error(self):
        """Test worker_strategy with rate limit error."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        swarm.mutex = asyncio.Lock()
        
        mock_push_socket = AsyncMock()
        error = Exception("429 Rate limit exceeded")
        
        result_message, keep_loop = await swarm.worker_strategy(
            mock_push_socket, None, error
        )
        
        assert result_message is None
        assert keep_loop is True


class TestGPTSwarmWorker:
    """Test GPTSwarm worker method."""
    
    @pytest.mark.asyncio
    async def test_worker_initialization(self):
        """Test worker method initialization."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        # Mock required async context manager attributes
        swarm.mutex = asyncio.Lock()
        swarm.tokens_status = asyncio.Event()
        swarm.tokens_status.set()  # Allow immediate execution
        swarm.nb_requests = 0
        swarm.period = 0.02
        
        # Mock the context and socket
        mock_ctx = Mock()
        mock_socket = Mock()
        mock_socket.connect = Mock()
        mock_socket.close = Mock()
        mock_ctx.socket.return_value = mock_socket
        swarm.ctx = mock_ctx
        
        # Mock worker_strategy to return immediately
        swarm.worker_strategy = AsyncMock(return_value=(None, False))
        
        # Mock OpenAI client
        swarm.openai_client = AsyncMock()
        swarm.openai_client.responses.create = AsyncMock()
        swarm.openai_client.responses.create.side_effect = Exception("Stop execution")
        
        messages = [Message(role=Role.USER, content="Test message")]
        
        # The worker should handle the exception and exit
        result = await swarm.worker("test-worker", 1, messages)
        
        # Verify socket operations
        mock_ctx.socket.assert_called_once_with(zmq.PUSH)
        mock_socket.connect.assert_called_once_with(swarm.collector_address)
        mock_socket.close.assert_called_once()


class TestGPTSwarmSwarm:
    """Test GPTSwarm swarm method (main orchestration)."""
    
    @pytest.mark.asyncio
    async def test_swarm_initialization(self):
        """Test swarm method initializes counters properly."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        # Mock async context manager setup
        swarm.mutex = asyncio.Lock()
        swarm.tokens_status = asyncio.Event()
        
        conversations = [
            [Message(role=Role.USER, content="Test 1")],
            [Message(role=Role.USER, content="Test 2")]
        ]
        
        # Mock the collector and worker methods to avoid actual ZMQ operations
        async def mock_collector():
            await asyncio.sleep(0.01)  # Brief delay
            
        async def mock_worker(worker_id, nb_retries, messages):
            return Message(role=Role.ASSISTANT, content=f"Response for {worker_id}")
        
        swarm.collector = mock_collector
        swarm.worker = mock_worker
        
        result = await swarm.swarm(conversations)
        
        # Verify initialization happened
        assert swarm.nb_requests == 0
        assert swarm.total_tokens == 0
        assert swarm.consumed_tokens == 0
        assert swarm.tokens_status.is_set()
        
        # Verify result has correct length
        assert len(result) == 2
        assert all(isinstance(msg, Message) for msg in result if msg is not None)


class TestGPTSwarmCollector:
    """Test GPTSwarm collector method."""
    
    @pytest.mark.asyncio
    async def test_collector_socket_setup(self):
        """Test collector method sets up ZMQ socket correctly."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        # Mock ZMQ context and socket
        mock_ctx = Mock()
        mock_socket = AsyncMock()
        mock_socket.bind = Mock()
        mock_socket.poll = AsyncMock(return_value=0)  # No messages available
        mock_socket.close = Mock()
        mock_ctx.socket.return_value = mock_socket
        swarm.ctx = mock_ctx
        
        # Mock required attributes
        swarm.mutex = asyncio.Lock()
        swarm.start_timer = asyncio.Event()
        swarm.tokens_status = asyncio.Event()
        swarm.nb_tokens_per_mn = 180_000
        swarm.model_token_size = 8192
        swarm.consumed_tokens = 0
        swarm.total_tokens = 0
        
        # Create a collector task with a short timeout
        collector_task = asyncio.create_task(swarm.collector())
        
        # Wait a very short time then cancel
        try:
            await asyncio.wait_for(collector_task, timeout=0.1)
        except asyncio.TimeoutError:
            collector_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass
        
        # Verify socket setup
        mock_ctx.socket.assert_called_once_with(zmq.PULL)
        mock_socket.bind.assert_called_once_with(swarm.collector_address)


class TestGPTSwarmStopSwarm:
    """Test GPTSwarm stop_swarm method."""
    
    def test_stop_swarm_cancels_worker_tasks(self):
        """Test stop_swarm cancels worker tasks."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        # Mock loop and tasks
        mock_loop = Mock()
        mock_task1 = Mock()
        mock_task1.get_name.return_value = "worker-123"
        mock_task1.cancel = Mock()
        
        mock_task2 = Mock()
        mock_task2.get_name.return_value = "collector-456"
        mock_task2.cancel = Mock()
        
        mock_task3 = Mock()
        mock_task3.get_name.return_value = "worker-789"
        mock_task3.cancel = Mock()
        
        with patch('asyncio.all_tasks', return_value=[mock_task1, mock_task2, mock_task3]):
            swarm.loop = mock_loop
            swarm.stop_swarm()
        
        # Should cancel only worker tasks
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_not_called()  # Not a worker task
        mock_task3.cancel.assert_called_once()


class TestGPTSwarmIntegration:
    """Integration tests for GPTSwarm functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_context_manager_flow(self):
        """Test full async context manager flow."""
        swarm = GPTSwarm("test-key", 180_000, 3000, 8192)
        
        # This test verifies the context manager works end-to-end
        async with swarm as active_swarm:
            assert active_swarm.mutex is not None
            assert active_swarm.tokens_status is not None
            assert active_swarm.start_timer is not None
            assert active_swarm.ctx is not None
            assert active_swarm.loop is not None
        
        # After exiting context, cleanup should have occurred
        # (We can't easily test cleanup without complex mocking)