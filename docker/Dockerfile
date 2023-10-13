FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Instal basic utilities
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.8-dev git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -U opencv-python
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
COPY docker/requirements.txt ./

RUN pip install -r requirements.txt

RUN wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
RUN gunzip /usr/local/bin/ninja.gz
RUN chmod a+x /usr/local/bin/ninja

ADD ./ /segmatron/
WORKDIR /segmatron/