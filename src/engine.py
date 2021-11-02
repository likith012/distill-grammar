from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, scheduler, device):

    model.train()
    total_train_loss = 0

    final_outputs = []
    final_targets = []

    for idx, data in tqdm(enumerate(data_loader), total = len(data_loader)):

        ids = data['ids']
        mask = data['mask']
        target = data['target']

        ids = ids.to(device, dtype = torch.long)
        mask = mask.to(device, dtype = torch.long)
        target = target.to(device, dtype = torch.float)
        
        optimizer.zero_grad()
        outputs = model(
            ids, 
            token_type_ids = None, 
            attention_mask = mask,
            return_dict=True
        )
        
        logits = outputs.logits
        loss = loss_fn(logits, target)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()

        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        final_targets.extend(target.cpu().detach().tolist())
        final_outputs.extend(torch.sigmoid(logits).cpu().detach().flatten().tolist())
    
    return np.array(final_outputs), final_targets, total_train_loss / len(data_loader)


def eval_fn(data_loader, model, device):

    model.eval()
    total_val_loss = 0

    final_outputs = []
    final_targets = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(data_loader), total = len(data_loader)):
            ids = data['ids']
            mask = data['mask']
            target = data['target']

            ids = ids.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            target = target.to(device, dtype = torch.float)
            
            outputs = model(
                ids, 
                token_type_ids = None, 
                attention_mask = mask, 
                return_dict=True
            )
            
            logits = outputs.logits
            loss = loss_fn(logits, target)

            total_val_loss += loss.item()
            
            final_targets.extend(target.cpu().detach().tolist())
            final_outputs.extend(torch.sigmoid(logits).cpu().detach().flatten().tolist())

    return np.array(final_outputs), final_targets, total_val_loss / len(data_loader)

