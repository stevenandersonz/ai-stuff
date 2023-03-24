# Tic Tac Toe Game with OpenAI GPT-3.5 Agent

This is a Tic Tac Toe game with an AI agent powered by OpenAI's GPT-3.5 language model. The game is built using JavaScript for the front end and a Bottle-based Python API for communication with the AI agent.

Installation
To run the game, you will need to install the following software:

- Python 3
- Bottle (Python web framework)
- openai

Once you have installed Python 3, you can install Bottle and openai using pip:

```sh
pip install bottle
pip install openai
```

To use the OpenAI GPT-3.5 agent, you will need an OpenAI API key. You can sign up for an API key on the OpenAI website: https://beta.openai.com/signup/

Usage
To start the game, you will need to run the Python API server:

```sh
python ./server/server.py
```

This will start the Bottle-based API server on port 8080.

Next, open the index.html file in your web browser to start the Tic Tac Toe game.

When it is the AI agent's turn to make a move, the JavaScript front end will send a POST request to the / route of the Python API server to get the AI agent's move. The API server will use the OpenAI API to generate a move based on the current state of the game, and will return the move as a JSON object.

The front end will then update the Tic Tac Toe board with the AI agent's move, and will display any trash talk from the AI agent in the chat box.

## Cost of running

You can run the server with either `gpt-3.5-turbo` or `davinci-003`

- `gpt-3.5-turbo` cost ~$0.002/game
- `davinci-003` cost ~$0.02/game
