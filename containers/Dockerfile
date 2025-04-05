#Python version being used for this project
FROM python:3.12.9

# Meta-data
LABEL maintainer="Carlos Eduardo Tussi Leite <carlos.leite@edu.dsti.institute>" \
      description="Docker container for training the model."


# Set the working directory to /training
WORKDIR /training_root

# Set the project root environment
ENV PYTHONPATH=/training_root

# Copy requirements.txt 
COPY requirements.txt ./
# Copy training and pipeline modules
COPY source/training ./source/training
COPY source/pipelines ./source/pipelines
COPY source/config.py ./source/config.py
# Copy model folder
COPY models/ ./models

# Install the required libraries
RUN   pip install --upgrade pip && pip --no-cache-dir install -r requirements.txt


# Run training module
CMD ["python", "-u", "-m" ,"source.training.main"]     