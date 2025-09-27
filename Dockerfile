FROM python:3.9.0

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY tracking.py .

# Command to run the application
CMD ["python", "tracking.py"]