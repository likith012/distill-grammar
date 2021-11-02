import sys
import os
import torch

class BERTDataset(torch.utils.data.Dataset):

    def __init__(self, sentence, target, config):
        super(BERTDataset, self).__init__()

        self.sentence= sentence
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):

        sentence = str(self.sentence[idx])

        inputs = self.tokenizer.encode_plus( 
            sentence,
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt',
            truncation = True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            "ids": ids[0],
            "mask": mask[0],
            "target": self.target[idx],
        }
