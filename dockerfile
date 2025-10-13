# Use a slim Python image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install git to clone pylinac repo
RUN apt update && apt install -y git

# Copy the requirements file and install dependencies
# This is done first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
	pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Streamlit app code
COPY . .

# Expose the default Streamlit port
EXPOSE 8502

# Set the entry point to run the Streamlit app
# Replace 'your_app.py' with the name of your main Streamlit file
ENTRYPOINT ["streamlit", "run", "menu_analyzer.py", "--server.port=8502", "--server.enableCORS=true", "--server.enableXsrfProtection=false"]
