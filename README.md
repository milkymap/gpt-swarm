# GPT Swarm

A high-performance Python implementation for running multiple OpenAI GPT models in parallel, designed for high-throughput applications. This project uses modern async/await patterns with asyncio and ZeroMQ for efficient inter-process communication, while intelligently handling OpenAI's rate limiting restrictions.

## Features

- **Concurrent Processing**: Run multiple GPT-4o model requests in parallel for maximum throughput
- **Intelligent Rate Limiting**: Automatically handles OpenAI's tokens-per-minute and requests-per-minute limits
- **Async Architecture**: Built with modern Python asyncio for non-blocking operations
- **ZeroMQ Communication**: Efficient message passing between workers and collectors
- **Clean Architecture**: Well-separated concerns between swarm logic and OpenAI client
- **Error Handling**: Robust retry logic and error recovery mechanisms
- **Resource Management**: Proper cleanup and resource management with async context managers

## Technologies Used

- **Python 3.12** - Latest Python with improved async performance
- **OpenAI API** - GPT-4o model access via official Python SDK
- **asyncio** - Asynchronous I/O and concurrency
- **ZeroMQ** - High-performance asynchronous messaging library
- **Pydantic** - Data validation and settings management
- **Click** - Command-line interface creation

## Requirements

- **Python 3.12+** - This project requires Python 3.12 or higher for optimal async performance
- **OpenAI API Key** - Valid OpenAI API key with access to GPT-4o model

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/milkymap/gpt-swarm.git
   cd gpt-swarm
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key and run**:
   ```bash
   export OPENAI_API_KEY=sk-your-api-key-here
   python main.py start-swarming
   ```

## Usage Example

The default example runs 32 concurrent conversations asking about the Big Bang theory:

```python
from libraries.swarming import GPTSwarm
from dataschema import Message, Role

async def run_model(openai_api_key: str):
    async with GPTSwarm(
        openai_api_key=openai_api_key, 
        nb_tokens_per_mn=180_000, 
        nb_requests_per_mn=3000, 
        model_token_size=8192
    ) as model:
        conversations = [
            [Message(role=Role.USER, content='Please explain me the big bang in simple terms')] 
            for _ in range(32)  # 32 parallel conversations
        ]
        
        responses = await model.swarm(conversations)
        for response in responses:
            print(response.content if response else "No response")
```

## Configuration

The GPTSwarm class accepts several configuration parameters for fine-tuning:

- `nb_tokens_per_mn`: Maximum tokens per minute (default: 180,000)
- `nb_requests_per_mn`: Maximum requests per minute (default: 3,000)
- `model_token_size`: Expected token size per model response (default: 8,192)

These values should be adjusted based on your OpenAI plan limits.

## Performance Notes

- **Python 3.12 Benefits**: This upgrade takes advantage of Python 3.12's improved async performance and better error handling
- **Concurrent Limits**: The system automatically manages rate limits to prevent API quota exhaustion
- **Memory Efficiency**: Uses async context managers for proper resource cleanup
- **Error Recovery**: Implements exponential backoff and retry logic for resilient operation

## Contributing

Contributions are welcome! Here's how to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows Python 3.12+ best practices and includes appropriate error handling.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### v2.0.0 (Latest)
- **Breaking**: Upgraded to Python 3.12+ requirement
- **Enhancement**: Updated README with comprehensive documentation
- **Enhancement**: Improved code examples and usage instructions
- **Enhancement**: Added performance notes and configuration details
