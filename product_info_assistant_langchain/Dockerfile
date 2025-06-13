.env 

# Use Python as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Start your Streamlit app
CMD ["streamlit", "run", "front_end/streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
