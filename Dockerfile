# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Added mirror for China since the user seems to be Chinese (based on "毕业设计")
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 and 8501 available to the world outside this container
EXPOSE 8000 8501

# Define environment variable
ENV PYTHONPATH=/app

# Run app.py when the container launches (Default, can be overridden)
CMD ["bash"]
