FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y python3 wget     

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda create -y -n torch python=3.7

WORKDIR /root

COPY . distill-grammer

RUN /bin/bash -c "cd distill-grammer \
    && source activate torch \
    && pip install -r requirements.txt"


