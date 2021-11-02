import transformers
import sys
import os
import torch

MAX_LEN = 64
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 10
ONNX_DEVICE = torch.device('cuda:0')

MODEL_NAME = 'bert-base-uncased'
TRAINING_FILE = os.path.join(sys.path[0], "input/cola_public/raw/in_domain_train.tsv")
MODEL_PATH = os.path.join(sys.path[0], "saves")

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')

    print(f"GPU available: {torch.cuda.get_device_name(0)}")

else:
    DEVICE = torch.device('cpu')
    print("No GPU available")

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    MODEL_NAME,
    do_lowercase = True
)

MODEL = transformers.BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = 1,
    output_attentions = False,
    output_hidden_states = False
    )