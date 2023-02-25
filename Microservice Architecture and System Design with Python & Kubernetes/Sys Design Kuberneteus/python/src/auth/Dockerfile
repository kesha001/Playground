# Base image python 3.10, the rest of image will be built on top of this image
FROM python:3.10-slim-bullseye

# Install OS dependencies, 
# additional flags (-y, -no-install-recommends, etc.) are to avoid installing unnecessary additional packages 
# and avoid taking up additional space with caching
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
    build-essential default-libmysqlclient-dev \
    && pip install --no-cache-dir --upgrade pip

# set working directory to /app
WORKDIR /app
# copy requirements file, we use it in separate layer so if code changes we can still use cached requirements
COPY ./requirements.txt /app
# install requirements
RUN pip install --no-cache-dir --requirement /app/requirements.txt
# then copy rest of application, the "." means current directory where we ran docker build command in on our local machine
COPY . /app

# app will run on port 5000
EXPOSE 5000

# create command to run when running our image
CMD ["python3", "server.py"]

