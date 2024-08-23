# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables to avoid buffering issues with logs
ENV PYTHONUNBUFFERED=1

# Create and set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 5000

# Set the command to run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "temple:app"]
