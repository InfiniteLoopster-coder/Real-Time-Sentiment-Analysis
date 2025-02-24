FROM python:3.8-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire codebase into the container
COPY . .

EXPOSE 5000

CMD ["python", "deployment/app.py"]
