# Real-Time Sentiment Analysis Pipeline for Social Media

## Overview
This project is an end-to-end MLOps pipeline that streams social media data (tweets), processes the data, fine-tunes a sentiment analysis model using a pre-trained BERT model, and deploys the model as a REST API. The project incorporates key MLOps principles including experiment tracking with MLflow, containerization with Docker, and optional orchestration with Kubernetes.

## Project Structure
Real-Time-Sentiment-Analysis/
│
├── data_ingestion/
│   ├── twitter_stream.py       # Code to stream tweets
│   └── config.py               # Twitter API credentials/config
│
├── data_processing/
│   └── preprocess.py           # Functions to clean and preprocess tweets
│
├── model_training/
│   ├── train.py                # Script for fine-tuning our model
│   └── model.py                # Model definition or utilities
│
├── deployment/
│   ├── app.py                  # Flask app to serve our model as an API
│   ├── Dockerfile              # Dockerfile to containerize the API
│   └── k8s/                    
│       ├── deployment.yaml     # Kubernetes deployment file
│       └── service.yaml        # Kubernetes service file
│
├── experiments/                # Folder for MLflow experiment logs
│
└── README.md                   # Project documentation

## Prerequisites
- **Python 3.8+**
- **Twitter API Credentials:**  
  Obtain your `API_KEY`, `API_SECRET`, `ACCESS_TOKEN`, and `ACCESS_TOKEN_SECRET` from the [Twitter Developer Platform](https://developer.twitter.com/).
- **Docker:** For containerizing the application.
- **Kubernetes (Optional):** For deploying the containerized app.
- **MLflow:** For tracking experiments (install via pip).

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your_username/Real-Time-Sentiment-Analysis.git
   cd Real-Time-Sentiment-Analysis

2. **Create and Activate a Virtual Environment:**
   ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

4. Configuration
Twitter API Credentials:
Update the credentials in data_ingestion/config.py (or directly in twitter_stream.py for testing) with your Twitter API keys.

MLflow Setup:
Ensure MLflow is installed and properly configured. The training script logs parameters and the trained model under the experiment "Sentiment Analysis Experiment".

5. Usage
1. Data Ingestion
Streaming Tweets: The data_ingestion/twitter_stream.py script uses Tweepy to stream tweets. Run it as follows:
    ```bash
    python data_ingestion/twitter_stream.py

2. Data Processing
Cleaning Tweets: The module in data_processing/preprocess.py provides functions to clean tweet text by removing URLs, mentions, hashtags, and extra spaces. Integrate this module into your data pipeline to preprocess tweets before training.
3. Model Training
Fine-Tuning BERT: The model_training/train.py script demonstrates how to fine-tune a pre-trained BERT model for sentiment analysis using the Hugging Face Transformers library. Run the script:
     ```bash
    python model_training/train.py

Notes:
Replace the dummy dataset with your actual preprocessed tweets and corresponding sentiment labels.
The script uses MLflow to log experiment parameters, metrics, and the final model.

4. Deployment
Flask API: The deployment/app.py file contains a Flask application that loads the trained model and exposes a /predict endpoint. Start the API with:
    ```bash
    python deployment/app.py

Docker Containerization: Build and run a Docker container using the provided Dockerfile:
    ```bash
    docker build -t your-dockerhub-username/sentiment-analysis .
    docker run -p 5000:5000 your-dockerhub-username/sentiment-analysis

Project Workflow
1.Data Ingestion:
Stream tweets in real-time using the Twitter API.
2.Data Processing:
Clean and preprocess the tweets.
3.Model Training:
Fine-tune a BERT model for sentiment analysis and log experiments with MLflow.
4.Deployment:
Deploy the trained model as a REST API using Flask.
5.Containerization & Orchestration (Optional):
Containerize with Docker and deploy using Kubernetes.


