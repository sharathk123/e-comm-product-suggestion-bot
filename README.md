# Project Overview

This project is a Flask-based web application developed for an e-commerce company. The application enables users to chat with an AI bot that provides product suggestions. It utilizes a Retrieval-Augmented Generation (RAG) architecture and integrates various advanced technologies including AstraDB as a vector database, LangChain, Groq for llama model, and OpenAI embedding.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Deployment](#deployment)


## Requirements
- Python 3.8+
- Flask 2.0+
- AstraDB (Vector Database)
- OpenAI API key
- LangChain library
- Llama 3.1 model (integrated via Groq)
- Docker (optional for containerized deployment)

## Installation
1. Clone the repository to your local environment:

    ```bash
    git clone https://github.com/sharathk123/e-comm-product-suggestion-bot.git
    cd e-comm-product-suggestion-bot
    ```

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration
Ensure that you have set up the following environment variables:

- `ASTRADB_URI` for the AstraDB connection URI
- `ASTRADB_TOKEN` for the AstraDB token
- `OPENAI_API_KEY` for OpenAI API access
- `GROQ_API_KEY` for Groq API access


Configuration files:

- `setup.py` for Flask and environment-specific configurations

## Running the Project
To run the application locally, use the following command:

```bash
flask run
```
The application will start on http://localhost:5000 by default.

## Running the Project on Render with Docker image
Read the docker-image.yaml in .github/workflows folder.
