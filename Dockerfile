# Use an appropriate base image with Python support
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set a working directory inside the container
WORKDIR /workspace

# Copy your shell script and Python scripts into the container
COPY run_sample_brain.sh /workspace/
COPY run_sample_abdom.sh /workspace/

ADD requirements.txt /workspace/
# ADD preprocess_mood.py /workspace/


RUN pip install -r /workspace/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# RUN chmod +x preprocess_mood.py

# Make the shell script executable
ADD models /workspace/models/
ADD scripts /workspace/scripts/

RUN chmod +x /workspace/*.sh

RUN mkdir /mnt/data
RUN mkdir /mnt/pred
ENV TMPDIR=/mnt/data
# ENV TMPDIR=/mnt/pred


# docker tag mood2023 docker.synapse.org/syn52393867/mood2023
# docker push docker.synapse.org/syn52393867/mood2023:latest

