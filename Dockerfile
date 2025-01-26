FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for Flask and Streamlit
EXPOSE 9696 8501

# Run both Flask and Streamlit in one container(makes the app accessible externally, not just localhost)
CMD ["sh", "-c", "python predict.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
