# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir waitress
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code to the container
COPY . .

# Expose a port (optional)
EXPOSE 8000

# Define the command to run your application
CMD ["waitress-serve", "--port=8000", "app:app"]
