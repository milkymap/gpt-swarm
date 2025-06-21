"""
Unit tests for dataschema module.
"""
import pytest
from pydantic import ValidationError

from dataschema import REQUEST_TYPE, Role, Message


class TestRequestType:
    """Test REQUEST_TYPE enum."""
    
    def test_request_type_values(self):
        """Test REQUEST_TYPE enum values."""
        assert REQUEST_TYPE.TIME2SLEEP == "time2sleep"
        assert REQUEST_TYPE.AVAILABLE_TOKEN == "available_token"
    
    def test_request_type_is_string_enum(self):
        """Test that REQUEST_TYPE is a string enum."""
        assert isinstance(REQUEST_TYPE.TIME2SLEEP, str)
        assert isinstance(REQUEST_TYPE.AVAILABLE_TOKEN, str)


class TestRole:
    """Test Role enum."""
    
    def test_role_values(self):
        """Test Role enum values."""
        assert Role.USER == "user"
        assert Role.SYSTEM == "system"
        assert Role.ASSISTANT == "assistant"
    
    def test_role_is_string_enum(self):
        """Test that Role is a string enum."""
        assert isinstance(Role.USER, str)
        assert isinstance(Role.SYSTEM, str)
        assert isinstance(Role.ASSISTANT, str)


class TestMessage:
    """Test Message pydantic model."""
    
    def test_message_creation_with_valid_data(self):
        """Test creating a message with valid data."""
        message = Message(role=Role.USER, content="Hello, world!")
        assert message.role == Role.USER
        assert message.content == "Hello, world!"
    
    def test_message_creation_with_all_roles(self):
        """Test creating messages with all role types."""
        user_msg = Message(role=Role.USER, content="User message")
        system_msg = Message(role=Role.SYSTEM, content="System message")
        assistant_msg = Message(role=Role.ASSISTANT, content="Assistant message")
        
        assert user_msg.role == Role.USER
        assert system_msg.role == Role.SYSTEM
        assert assistant_msg.role == Role.ASSISTANT
    
    def test_message_creation_with_empty_content(self):
        """Test creating a message with empty content."""
        message = Message(role=Role.USER, content="")
        assert message.role == Role.USER
        assert message.content == ""
    
    def test_message_creation_with_long_content(self):
        """Test creating a message with long content."""
        long_content = "This is a very long message content. " * 100
        message = Message(role=Role.USER, content=long_content)
        assert message.role == Role.USER
        assert message.content == long_content
    
    def test_message_validation_missing_role(self):
        """Test message validation fails when role is missing."""
        with pytest.raises(ValidationError):
            Message(content="Hello, world!")
    
    def test_message_validation_missing_content(self):
        """Test message validation fails when content is missing."""
        with pytest.raises(ValidationError):
            Message(role=Role.USER)
    
    def test_message_validation_invalid_role(self):
        """Test message validation fails with invalid role."""
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="Hello, world!")
    
    def test_message_serialization(self):
        """Test message serialization to dict."""
        message = Message(role=Role.USER, content="Test content")
        message_dict = message.model_dump()
        
        expected = {
            "role": "user",
            "content": "Test content"
        }
        assert message_dict == expected
    
    def test_message_deserialization(self):
        """Test message deserialization from dict."""
        data = {
            "role": "assistant",
            "content": "AI response"
        }
        message = Message(**data)
        
        assert message.role == Role.ASSISTANT
        assert message.content == "AI response"
    
    def test_message_equality(self):
        """Test message equality comparison."""
        msg1 = Message(role=Role.USER, content="Same content")
        msg2 = Message(role=Role.USER, content="Same content")
        msg3 = Message(role=Role.ASSISTANT, content="Same content")
        msg4 = Message(role=Role.USER, content="Different content")
        
        assert msg1 == msg2
        assert msg1 != msg3
        assert msg1 != msg4
    
    def test_message_string_representation(self):
        """Test message string representation."""
        message = Message(role=Role.USER, content="Test message")
        str_repr = str(message)
        
        assert "role='user'" in str_repr or "role=<Role.USER:" in str_repr
        assert "content='Test message'" in str_repr