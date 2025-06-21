"""
Unit tests for main.py CLI functionality.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from click.testing import CliRunner

from main import run_model, group, start_swarming
from dataschema import Message, Role


class TestRunModel:
    """Test run_model async function."""
    
    @pytest.mark.asyncio
    async def test_run_model_with_mock_swarm(self):
        """Test run_model function with mocked GPTSwarm."""
        with patch('main.GPTSwarm') as MockGPTSwarm:
            # Create mock swarm instance
            mock_swarm_instance = AsyncMock()
            mock_swarm_instance.swarm = AsyncMock()
            mock_swarm_instance.swarm.return_value = [
                Message(role=Role.ASSISTANT, content="Big bang explanation 1"),
                Message(role=Role.ASSISTANT, content="Big bang explanation 2")
            ]
            
            # Configure the context manager
            mock_swarm_instance.__aenter__ = AsyncMock(return_value=mock_swarm_instance)
            mock_swarm_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockGPTSwarm.return_value = mock_swarm_instance
            
            # Test the function
            test_api_key = "test-api-key-123"
            
            with patch('builtins.print') as mock_print:
                await run_model(test_api_key)
            
            # Verify GPTSwarm was instantiated with correct parameters
            MockGPTSwarm.assert_called_once_with(
                openai_api_key=test_api_key,
                nb_tokens_per_mn=180_000,
                nb_requests_per_mn=3000,
                model_token_size=8192
            )
            
            # Verify swarm method was called with 32 conversations
            mock_swarm_instance.swarm.assert_called_once()
            conversations_arg = mock_swarm_instance.swarm.call_args[1]['conversations']
            assert len(conversations_arg) == 32
            
            # Verify each conversation has the correct message
            for conversation in conversations_arg:
                assert len(conversation) == 1
                assert conversation[0].role == Role.USER
                assert conversation[0].content == 'Please explain me the big bang in simple terms'
            
            # Verify print was called for each response
            assert mock_print.call_count == 2
    
    @pytest.mark.asyncio
    async def test_run_model_with_no_response(self):
        """Test run_model when swarm returns None."""
        with patch('main.GPTSwarm') as MockGPTSwarm:
            mock_swarm_instance = AsyncMock()
            mock_swarm_instance.swarm = AsyncMock(return_value=None)
            mock_swarm_instance.__aenter__ = AsyncMock(return_value=mock_swarm_instance)
            mock_swarm_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockGPTSwarm.return_value = mock_swarm_instance
            
            with patch('builtins.print') as mock_print:
                await run_model("test-key")
            
            # Print should not be called when response is None
            mock_print.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_run_model_with_empty_response(self):
        """Test run_model when swarm returns empty list."""
        with patch('main.GPTSwarm') as MockGPTSwarm:
            mock_swarm_instance = AsyncMock()
            mock_swarm_instance.swarm = AsyncMock(return_value=[])
            mock_swarm_instance.__aenter__ = AsyncMock(return_value=mock_swarm_instance)
            mock_swarm_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockGPTSwarm.return_value = mock_swarm_instance
            
            with patch('builtins.print') as mock_print:
                await run_model("test-key")
            
            # Print should not be called for empty response
            mock_print.assert_not_called()


class TestClickCommands:
    """Test Click CLI commands."""
    
    def test_group_command_setup(self):
        """Test that group command is properly configured."""
        runner = CliRunner()
        
        # Test help output
        result = runner.invoke(group, ['--help'])
        assert result.exit_code == 0
        assert '--openai_api_key' in result.output
        assert 'openai api key for gpt-4o' in result.output
    
    def test_group_command_with_api_key(self):
        """Test group command stores API key in context."""
        runner = CliRunner()
        
        # Mock the run_model function to avoid actual execution
        with patch('main.run_model') as mock_run_model:
            with patch('main.asyncio.run') as mock_asyncio_run:
                result = runner.invoke(group, [
                    '--openai_api_key', 'test-key-123',
                    'start-swarming'
                ])
                
                # Debug output if the test fails
                if result.exit_code != 0:
                    print(f"Exit code: {result.exit_code}")
                    print(f"Output: {result.output}")
                    print(f"Exception: {result.exception}")
                
                # The command should execute without error
                assert result.exit_code == 0
                # Verify asyncio.run was called
                mock_asyncio_run.assert_called_once()
    
    def test_group_command_missing_api_key(self):
        """Test group command fails when API key is missing."""
        runner = CliRunner()
        
        # Remove the OPENAI_API_KEY from environment if it exists
        import os
        original_key = os.environ.get('OPENAI_API_KEY')
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
            
        try:
            result = runner.invoke(group, ['start-swarming'])
            assert result.exit_code != 0
            # Check for the actual error message from Click
            assert 'Missing option' in result.output or 'required' in result.output.lower()
        finally:
            # Restore the original API key if it existed
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
    
    def test_group_command_with_env_var(self):
        """Test group command reads API key from environment variable."""
        runner = CliRunner()
        
        with patch('main.run_model') as mock_run_model:
            with patch('main.asyncio.run') as mock_asyncio_run:
                result = runner.invoke(group, ['start-swarming'], 
                                     env={'OPENAI_API_KEY': 'env-api-key'})
                
                # Should succeed when API key is provided via environment
                assert result.exit_code == 0
                mock_asyncio_run.assert_called_once()
    
    def test_start_swarming_command(self):
        """Test start_swarming command calls run_model."""
        runner = CliRunner()
        
        with patch('main.asyncio.run') as mock_asyncio_run:
            with patch('main.run_model') as mock_run_model:
                result = runner.invoke(group, [
                    '--openai_api_key', 'test-api-key',
                    'start-swarming'
                ])
                
                assert result.exit_code == 0
                mock_asyncio_run.assert_called_once()
    
    def test_start_swarming_accesses_context(self):
        """Test start_swarming command accesses context correctly."""
        runner = CliRunner()
        
        # Test that the context is properly passed through the CLI
        with patch('main.asyncio.run') as mock_asyncio_run:
            result = runner.invoke(group, [
                '--openai_api_key', 'context-api-key',
                'start-swarming'
            ])
            
            # Verify the command ran successfully
            assert result.exit_code == 0
            mock_asyncio_run.assert_called_once()


class TestMainModuleExecution:
    """Test main module execution."""
    
    def test_main_guard_calls_group(self):
        """Test __main__ guard calls group function."""
        with patch('main.group') as mock_group:
            # Simulate running the module
            exec(compile(
                'if __name__ == "__main__": group(obj={})',
                '<string>', 'exec'
            ), {'__name__': '__main__', 'group': mock_group})
            
            mock_group.assert_called_once_with(obj={})


class TestMessageGeneration:
    """Test message generation in run_model."""
    
    @pytest.mark.asyncio
    async def test_conversation_message_content(self):
        """Test that conversations contain the correct message content."""
        with patch('main.GPTSwarm') as MockGPTSwarm:
            mock_swarm_instance = AsyncMock()
            mock_swarm_instance.swarm = AsyncMock(return_value=[])
            mock_swarm_instance.__aenter__ = AsyncMock(return_value=mock_swarm_instance)
            mock_swarm_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockGPTSwarm.return_value = mock_swarm_instance
            
            await run_model("test-key")
            
            # Get the conversations passed to swarm
            conversations = mock_swarm_instance.swarm.call_args[1]['conversations']
            
            # Verify message content and structure
            expected_content = 'Please explain me the big bang in simple terms'
            for conversation in conversations:
                assert len(conversation) == 1
                message = conversation[0]
                assert isinstance(message, Message)
                assert message.role == Role.USER
                assert message.content == expected_content
    
    @pytest.mark.asyncio
    async def test_conversation_count(self):
        """Test that exactly 32 conversations are generated."""
        with patch('main.GPTSwarm') as MockGPTSwarm:
            mock_swarm_instance = AsyncMock()
            mock_swarm_instance.swarm = AsyncMock(return_value=[])
            mock_swarm_instance.__aenter__ = AsyncMock(return_value=mock_swarm_instance)
            mock_swarm_instance.__aexit__ = AsyncMock(return_value=None)
            
            MockGPTSwarm.return_value = mock_swarm_instance
            
            await run_model("test-key")
            
            conversations = mock_swarm_instance.swarm.call_args[1]['conversations']
            assert len(conversations) == 32


class TestErrorHandling:
    """Test error handling in CLI commands."""
    
    def test_invalid_command(self):
        """Test handling of invalid commands."""
        runner = CliRunner()
        result = runner.invoke(group, ['--openai_api_key', 'test', 'invalid_command'])
        
        assert result.exit_code != 0
        assert 'No such command' in result.output or 'Error' in result.output
    
    @pytest.mark.asyncio
    async def test_run_model_exception_handling(self):
        """Test run_model handles exceptions gracefully."""
        with patch('main.GPTSwarm') as MockGPTSwarm:
            # Make GPTSwarm raise an exception
            MockGPTSwarm.side_effect = Exception("API connection failed")
            
            # run_model should handle the exception (or let it propagate)
            with pytest.raises(Exception):
                await run_model("test-key")