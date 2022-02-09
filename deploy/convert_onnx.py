"""Convert the model to ONN format
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"


import torch
import torch.nn as nn
import torch.onnx
import sys, os

sys.path.insert(0, os.path.join(sys.path[0], '../'))

from configs import config
from src.dataset import BERTDataset


if __name__ == '__main__':

    sentence = ['I love BERT']

    dataset = BERTDataset(sentence = sentence, target = [1], config = config)

    model = config.MODEL

    num_device = torch.cuda.device_count()
    device_ids = list(range(num_device))
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model = model.module if hasattr(model, 'module') else model
    model = config.MODEL.from_pretrained(config.MODEL_PATH, local_files_only = True)
    model.eval()

    ids = dataset[0]['ids'].unsqueeze(0)
    attention_mask = dataset[0]['mask'].unsqueeze(0)
    token_type_ids = None

    device = 'cpu'

    ids = ids.to(device, dtype = torch.long)
    attention_mask = attention_mask.to(device, dtype = torch.long)

    torch.onnx.export(
        model,
        (ids, token_type_ids, attention_mask),
        "onnx_model.onnx",
        input_names = ['ids', 'token_type_ids' 'attention_mask'],
        output_names = ['output'],
        dynamic_axes = {
            'ids': {0: 'batch_size'},
            'token_type_ids': {0, 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'output': {0: 'batch_size'},
    },
        verbose = True,
        opset_version=12,  
        enable_onnx_checker=True
    )