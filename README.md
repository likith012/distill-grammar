[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# distill-grammar

## Introduction 🔥

>An English language grammar classification model trained on Bert. The self-supervised pretrained model is a Bert-base model trained on large corpus of 
text data. Here we explore the possibility of few shot learning by using the pretrained model to classify the text data. The model is trained on the **cola public** dataset. It's a dataset that contains a very few labelled sentences from the public domain in English. 

## Results :man_dancing:

> Performance metrics

|          | MCC Score | Accuracy | 
| -------- | ------------- | ------------- |
| Train | 0.9951 | 0.998 | 
| Test| 0.5699 | 0.8299 | 

