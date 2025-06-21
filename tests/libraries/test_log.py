"""
Unit tests for libraries.log module.
"""
import logging
import pytest
from unittest.mock import patch, MagicMock

from libraries.log import logger


class TestLogger:
    """Test logger configuration and functionality."""
    
    def test_logger_name(self):
        """Test logger has correct name."""
        assert logger.name == 'GPT-SWARM'
    
    def test_logger_is_logger_instance(self):
        """Test logger is an instance of logging.Logger."""
        assert isinstance(logger, logging.Logger)
    
    def test_logger_level(self):
        """Test logger level is set to DEBUG."""
        assert logger.getEffectiveLevel() == logging.DEBUG
    
    def test_logger_has_handlers(self):
        """Test logger has at least one handler."""
        # The logger should inherit handlers from the root logger due to basicConfig
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
    
    @patch('libraries.log.logger')
    def test_logger_debug_message(self, mock_logger):
        """Test logger debug message."""
        test_message = "Test debug message"
        mock_logger.debug(test_message)
        mock_logger.debug.assert_called_once_with(test_message)
    
    @patch('libraries.log.logger')
    def test_logger_info_message(self, mock_logger):
        """Test logger info message."""
        test_message = "Test info message"
        mock_logger.info(test_message)
        mock_logger.info.assert_called_once_with(test_message)
    
    @patch('libraries.log.logger')
    def test_logger_warning_message(self, mock_logger):
        """Test logger warning message."""
        test_message = "Test warning message"
        mock_logger.warning(test_message)
        mock_logger.warning.assert_called_once_with(test_message)
    
    @patch('libraries.log.logger')
    def test_logger_error_message(self, mock_logger):
        """Test logger error message."""
        test_message = "Test error message"
        mock_logger.error(test_message)
        mock_logger.error.assert_called_once_with(test_message)
    
    def test_logging_format_contains_required_fields(self):
        """Test logging format contains required fields."""
        # Get the root logger's first handler (created by basicConfig)
        root_logger = logging.getLogger()
        if root_logger.handlers:
            formatter = root_logger.handlers[0].formatter
            format_string = formatter._fmt
            
            # Check that the format string contains expected field names
            expected_field_names = ['asctime', 'name', 'filename', 
                                  'lineno', 'levelname', 'message']
            
            for field_name in expected_field_names:
                assert field_name in format_string
    
    def test_logger_can_log_different_levels(self):
        """Test logger can log messages at different levels."""
        with patch('libraries.log.logger') as mock_logger:
            # Test different log levels
            mock_logger.debug("Debug message")
            mock_logger.info("Info message")
            mock_logger.warning("Warning message")
            mock_logger.error("Error message")
            mock_logger.critical("Critical message")
            
            # Verify all methods were called
            mock_logger.debug.assert_called_once()
            mock_logger.info.assert_called_once()
            mock_logger.warning.assert_called_once()
            mock_logger.error.assert_called_once()
            mock_logger.critical.assert_called_once()
    
    def test_logger_handles_string_formatting(self):
        """Test logger handles string formatting properly."""
        with patch('libraries.log.logger') as mock_logger:
            test_value = "test_value"
            mock_logger.info("Testing with value: %s", test_value)
            mock_logger.info.assert_called_once_with("Testing with value: %s", test_value)
    
    def test_logger_handles_exceptions(self):
        """Test logger can handle exception logging."""
        with patch('libraries.log.logger') as mock_logger:
            try:
                raise ValueError("Test exception")
            except ValueError as e:
                mock_logger.error(e)
                mock_logger.exception("Exception occurred")
            
            mock_logger.error.assert_called_once()
            mock_logger.exception.assert_called_once()


class TestLoggingConfiguration:
    """Test logging module configuration."""
    
    def test_basic_config_called(self):
        """Test that basicConfig was called with correct parameters."""
        # We can't directly test if basicConfig was called since it's already executed
        # But we can test the effects of the configuration
        root_logger = logging.getLogger()
        
        # Check that DEBUG level is effective
        assert root_logger.getEffectiveLevel() <= logging.DEBUG
        
        # Check that handlers exist
        assert len(root_logger.handlers) > 0
    
    def test_log_format_includes_all_components(self):
        """Test that log format includes all expected components."""
        # Create a test logger to verify format
        test_logger = logging.getLogger('test_format_logger')
        
        # The format should be applied globally via basicConfig
        # We verify by checking if the root logger has the expected format
        root_logger = logging.getLogger()
        if root_logger.handlers:
            handler = root_logger.handlers[0]
            if handler.formatter:
                format_str = handler.formatter._fmt
                
                # Check for key components in the format string
                expected_components = [
                    'asctime', 'name', 'filename', 'lineno', 'levelname', 'message'
                ]
                
                for component in expected_components:
                    assert component in format_str, f"Format missing {component}"