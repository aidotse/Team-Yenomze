# set base image (host OS)
FROM nvcr.io/nvidia/pytorch:20.10-py3

# set the working directory in the container
WORKDIR /workspace

# copy the dependencies file to the working directory
COPY requirements.txt /tmp/requirements.txt

# install dependencies
RUN pip install -r /tmp/requirements.txt
