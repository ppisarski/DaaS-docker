FROM python:buster
MAINTAINER Pawel Pisarski <mail@pawelpisarski.com>

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR  /usr/src/Titanic

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
RUN ls -la /src/*

ENV PYTHONPATH "${PYTONPATH}:/"

# Running Python Application
CMD ["python3", "/src/main.py"]
