import sys
import os
import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
import wandb
import numpy as np
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(sys.path[0], '../'))

from src import dataset, engine
from configs import config

def run(wandb):

    df = pd.read_csv(config.TRAINING_FILE, delimiter = '\t', header = None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size = 0.2,
        random_state = 42,
        stratify = df.label.values
    )

    # Drop indices
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        sentence = df_train.sentence.values,
        target = df_train.label.values,
        config = config
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        sampler = torch.utils.data.RandomSampler(train_dataset)
    )

    valid_dataset = dataset.BERTDataset(
        sentence = df_valid.sentence.values,
        target = df_valid.label.values,
         config = config
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        sampler = torch.utils.data.SequentialSampler(valid_dataset)
    )

    model = config.MODEL.to(config.DEVICE)

    param_list = list(model.named_parameters()) # list of model parameters
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'] # bias of hidden layer and layer norm are not decayed
    optimizer_parameters = [ # removing the no_decay params
        {'params': [param for name, param in param_list if not any(nd in name for nd in no_decay)], 'weight_deacy': 0.001}, # without no_decay params
        {'params': [param for name, param in param_list if any(nd in name for nd in no_decay)], 'weight_deacy': 0.} # only no_decay params
    ]

    num_train_steps = int((len(df_train) / config.TRAIN_BATCH_SIZE) * config.EPOCHS)

    optimizer = AdamW(optimizer_parameters, lr = 3e-5, eps = 1e-8)
    scheduler = get_linear_schedule_with_warmup( # gets called in train_fn as this scheduler is independant on valid loss
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps
    )

    num_device = torch.cuda.device_count()
    device_ids = list(range(num_device))
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids) # distributed training

    best_accuracy = 0

    for epoch in range(config.EPOCHS):
        
        print(f'Epoch: {epoch}', '\n')
        train_outputs, train_targets, train_loss = engine.train_fn(train_dataloader, model, optimizer, scheduler, config.DEVICE)
        valid_outputs, valid_targets, valid_loss = engine.eval_fn(valid_dataloader, model, config.DEVICE)

        wandb.log({'Train loss': train_loss, 'Epoch': epoch})
        wandb.log({'Valid loss': valid_loss, 'Epoch': epoch})

        train_outputs = train_outputs >= 0.5

        train_accuracy = metrics.accuracy_score(train_targets, train_outputs)
        train_mcc = metrics.matthews_corrcoef(train_targets, train_outputs)
        print(f"Train Accuracy Score: {train_accuracy}")
        wandb.log({'Train Accuracy Score': train_accuracy, 'Epoch': epoch})
        print(f"Train MCC Score: {train_mcc}")
        wandb.log({'Train MCC Score': train_mcc, 'Epoch': epoch})

        valid_outputs = valid_outputs >= 0.5

        accuracy = metrics.accuracy_score(valid_targets, valid_outputs)
        valid_mcc = metrics.matthews_corrcoef(valid_targets, valid_outputs)
        print(f"Valid Accuracy Score: {accuracy}")
        wandb.log({'Valid Accuracy Score': accuracy, 'Epoch': epoch})
        print(f"Valid MCC Score: {valid_mcc}")
        wandb.log({'Valid MCC Score': valid_mcc, 'Epoch': epoch})

        if not os.path.exists(config.MODEL_PATH):
            os.mkdir(config.MODEL_PATH)

        if accuracy > best_accuracy:

            saved_model = model.module if hasattr(model, 'module') else model
            saved_model.save_pretrained(config.MODEL_PATH)
            config.TOKENIZER.save_pretrained(config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == '__main__':

    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    wandb.init()
    wandb_config = wandb.config
    wandb_config.epochs = config.EPOCHS
    wandb_config.train_batch_size = config.TRAIN_BATCH_SIZE
    wandb_config.valid_batch_size = config.VALID_BATCH_SIZE
    wandb_config.max_len = config.MAX_LEN

    run(wandb)
