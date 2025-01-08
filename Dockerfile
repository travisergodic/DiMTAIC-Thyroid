# Use the official PyTorch image with PyTorch 2.0.0 and CUDA 11.7
# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# FROM intel/intel-optimized-pytorch:2.0.0-idp-base
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.12-py3.9.12-cuda11.3.1-u20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install any additional system dependencies if required
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt update && apt install -y zip unzip curl

# Install pip for Python 3.9 (if needed)
# RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py

# Define a working directory
WORKDIR /app

# Copy your project files into the container
COPY . /app

# RUN pip install --upgrade pip && pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# Remove all files in the pretrained/ directory
RUN rm -rf pretrained/*

# Download the file using wget and save it in the pretrained/ directory
# RUN wget -O pretrained/segformer.b2.ade.pth https://travisergodic-ai-models.oss-cn-shanghai.aliyuncs.com/tianchi_thyroid/segformer.b2.ade.pth
RUN wget -O pretrained/pvt_v2_b2.pth https://travisergodic-ai-models.oss-cn-shanghai.aliyuncs.com/tianchi_thyroid/pvt_v2_b2.pth
# RUN wget -O pretrained/resnet50-19c8e357.pth https://travisergodic-ai-models.oss-cn-shanghai.aliyuncs.com/tianchi_thyroid/resnet50-19c8e357.pth
RUN pip install --no-cache-dir -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# Make your bash script executable
RUN chmod +x run.sh

# Run your service with run.sh or another entry point
CMD ["./run.sh"]