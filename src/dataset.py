"""Dataloader for the dataset.
"""

__author__ = "Likith Reddy"
__version__ = "1.0.0"
__email__ = "likith012@gmail.com"

import torch

class BERTDataset(torch.utils.data.Dataset):
    """DataLoader for the dataset.

    Attributes:
    ----------
    sentence : list[str]
        Input sentences.
    target : list[int]
        Target labels.
    tokenizer : transformers.BertTokenizer
        Tokenizer for the BERT model.
    max_len : int
        Maximum length of the input sentence.

    """

    def __init__(self, sentence: list[str], target: list[str], config):
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
