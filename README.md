[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# distill-grammar
#### An English language grammar classification model trained on Bert.

## Table of contents
- [Introduction](https://github.com/likith012/distill-grammar/edit/main/README.md#introduction-)
- [Results](https://github.com/likith012/distill-grammar/edit/main/README.md#results-man_dancing)
- [Getting started](https://github.com/likith012/distill-grammar/edit/main/README.md#getting-started-)


## Introduction 🔥

>An English language grammar classification model trained on Bert. The self-supervised pretrained model is a Bert-base model trained on large corpus of 
text data. Here we explore the possibility of few shot learning by using the pretrained model to classify the text data. The model is trained on the **cola public** dataset. It's a dataset that contains a very few labelled sentences from the public domain in English. 

## Results :man_dancing:

> Performance metrics

|          | MCC Score | Accuracy | 
| -------- | ------------- | ------------- |
| **Results**| 0.5699 | 0.8299 | 

## Getting started 🥷
#### Setting up the environment
- All the development work is done using `Python 3.7`
- Install all the necessary dependencies using `requirements.txt` file. Run `pip install -r requirements.txt` in terminal
- Alternatively, set up the environment and train the model using the `Dockerfile`. Run `docker build -f Dockerfile -t <image_name> .`

#### What each file does

- `configs/config.py` : This file contains all the configurations for the model.
- `src/dataset.py` : This file contains the utility functions for loading the dataset.
- `src/engine.py` : This file contains the utilities for training and evaluation.
- `src/train.py` : This file is used to train the model.
- `deploy/flask_app.py` : This file is used to deploy the model.
- `deploy/convert_onnx.py` : This file is used to convert the model to onnx format.
- `saves` : This directory contains bert tokenizer configuration, special tokens etc.

#### Training the model
- Run `python src/train.py` in terminal.
