# GPT-3 Swarm

This project provides a Python implementation for running multiple OpenAI's GPT-3 models in parallel, which is particularly useful for high-throughput applications. The implementation uses asyncio and ZeroMQ to communicate between the workers and the collector, and handles the rate-limiting restrictions imposed by OpenAI. The code provides a clean separation of concerns, with the swarm logic and the GPT-3 client being in different classes. The project also includes a sample client that demonstrates how to use the implementation to generate responses to multiple messages concurrently.

## Features

- Supports running multiple GPT-3 models concurrently.
- Uses asyncio and ZeroMQ for efficient communication between the workers and the collector.
- Handles the rate-limiting restrictions imposed by OpenAI.
- Provides a clean separation of concerns between the swarm logic and the GPT-3 client.
- Includes a sample client that demonstrates how to use the implementation.

## Technologies used

- Python 3.8
- OpenAI API
- asyncio
- ZeroMQ

## Installation

- Clone the repository.
- Install the dependencies using pip.
- Set the OpenAI API key in the configuration file.
- Run the sample client to generate responses to multiple messages concurrently.

```bash
python -m venv env 
source env/bin/activate 
pip install -r requirements.txt
export OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXX; python main.py start-swarming
```

## Contributing

Contributions are welcome! Please open an issue or pull request if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.
