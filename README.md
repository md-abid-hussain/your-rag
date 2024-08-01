# Simple RAG application: your-rag

This repository demonstrates a Retrieval-Augmented Generation (RAG) application built with LangChain, Streamlit, and Google Gemini. The app showcases the ease of deploying such applications to a Kubernetes cluster using the Cyclops UI.

## Features

- Interactive interface built with Streamlit
- Leverages Google Gemini for text generation
- Allows user to input text prompts Generates responses based on the prompt and retrieved information

## Running the App Locally

#### Prerequisites:

    Python 3.10+
    pip

#### 1. Create and Activate a Virtual Environment:

```bash
python -m venv .venv
source .venv/bin/activate  # For Linux/macOS
.venv\Scripts\activate  # For Windows

```

#### 2. Install Dependencies:

```bash
pip install -r requirements.txt
```

#### 3. Set Up Environment Variables:

- Create a file named `.env` in the project root directory.
- Copy the contents of `.env.example` to `.env` and update the `GOOGLE_API_KEY` variable with your actual API key.

#### 4. Run the App:

```bash
streamlit run main.py
```

## Run using Docker image

The provided Docker image allows for containerized deployment.
Docker image link: [DockerHub](https://hub.docker.com/r/mdabidhussain/your-rag)

#### 1. Pull the Image:

```bash
docker pull mdabidhussain/your-rag:latest
```

#### 2. Run the App with Docker:

```bash
docker run -e GOOGLE_API_KEY=your_api_key -p 8501:8501 mdabidhussain/your-rag:latest
```

This command will run the containerized app, exposing port 8501 on the host machine. Access the app at http://localhost:8501.

## Deployment via Cyclops UI

For a detailed walkthrough of the deployment process, check out this blog post where we used this exact application as a case study:

[From Docker to Kubernetes: A Journey with Cyclops UI](https://dev.to/mdabidhussain/from-docker-to-kubernetes-a-journey-with-cyclops-ui-m9l)

This blog post provides step-by-step instructions, screenshots, and best practices for deploying your application using Cyclops UI.

**Key benefits of using Cyclops UI:**

- Rapid deployment and management of Kubernetes applications
- User-friendly interface for non-technical users
- Streamlined configuration and deployment process
