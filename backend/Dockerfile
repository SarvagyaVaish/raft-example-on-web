# syntax=docker/dockerfile:1

# Start with the pytorch image
FROM pytorch/pytorch:latest as base

# Install some basic packages
RUN apt-get update
RUN apt-get install -y git wget zip unzip vim

# Install CVML packages
RUN apt-get install -y libgl1-mesa-glx
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libglib2.0-0
RUN apt-get install libxcb-xinerama0

# Install python packages
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt

# Clone the RAFT repo and download the models
# Reference the readme from learnopencv for more details
#   https://github.com/spmallick/learnopencv/tree/master/Optical-Flow-Estimation-using-Deep-Learning-RAFT
RUN git clone https://github.com/MaximKuklin/RAFT.git
# RUN git clone https://github.com/princeton-vl/RAFT.git
RUN ./RAFT/download_models.sh && rm models.zip

# Copy in the source code
COPY ./app /code/app

# Run the webserver
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

#
# Usage:
#   - Install gcloud
#   - Configure Docker permissions: gcloud auth configure-docker us-central1-docker.pkg.dev
#
# docker build -t raft-web-backend .
# docker run --rm -p 8080:8080 raft-web-backend
# docker tag raft-web-backend us-central1-docker.pkg.dev/raft-example/raft-repo/raft-web-backend:dev
# docker push us-central1-docker.pkg.dev/raft-example/raft-repo/raft-web-backend:dev
