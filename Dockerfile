FROM python:3.6-buster
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]