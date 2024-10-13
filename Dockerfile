# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure credentials.ini file is copied
COPY credentials.ini /app/credentials.ini

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "RAG.py", "--server.port=8501", "--server.enableCORS=false"]
