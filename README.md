# RAG-BOt

## Overview
RAG BOT is a document processing application that allows users to upload PDF files, extract and clean text, and interact with the content using a Generative AI model. It supports multiple queries and is designed to handle large documents efficiently.

## Table of Contents
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Running with Docker](#running-with-docker)
- [Usage](#usage)


## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Docker (if you plan to run the application using Docker)

### Install Required Packages
Before running the application, install the necessary Python packages by executing:


## Running Locally

1. **Clone the Repository**  
   Clone this repository to your local machine:

2. **Set Up API Key**  
Create a `credentials.ini` file in the project root directory with the following content:
Replace `YOUR_GOOGLE_API_KEY` with your actual Google API key.

3. **Run the Application**  
Start the application using Streamlit:

4. **Access the Application**  
Open your web browser and go to `http://localhost:8501` to access the application.

## Running with Docker

1. **Build the Docker Image**  
Navigate to the project directory and build the Docker image:
docker build -t rag-bot .


2. **Run the Docker Container**  
Run the Docker container, mapping port 8501 from the container to port 8501 on your host machine:
docker run -p 8501:8501 rag-bot

3. **Access the Application**  
Open your web browser and go to `http://localhost:8501` to access the application.

## Usage

- Upload multiple PDF files using the file uploader.
- Click the "Process Documents" button to extract and clean the text from the uploaded PDFs.
- Use the default questions provided or enter your own query to interact with the content.
- The application leverages a Generative AI model to generate responses based on the uploaded documents.


