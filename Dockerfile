# Step 1: Use a lightweight, official Python image as a base
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Step 4: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application files into the container
# This includes app.py, and the model/, static/, and templates/ folders
COPY . .

# Step 6: Expose the port that the application will run on
EXPOSE 80

# Step 7: Define the command to start the Uvicorn server when the container starts
# The --host 0.0.0.0 flag makes it accessible from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]