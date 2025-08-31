# Official Python 3.12 image
FROM python:3.12

# Set the working directory 
WORKDIR /app

# Add app.py and models directory
COPY app.py .
COPY models/ ./models/

# Install the required libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Specify default commands
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]