FROM python:3.10-slim
 
# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
 
WORKDIR /app
 
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy the entire project
COPY . .
 
# Generate the datasets at build time
RUN python generate_data.py

CMD ["openenv", "serve", "--host", "0.0.0.0", "--port", "7860"]