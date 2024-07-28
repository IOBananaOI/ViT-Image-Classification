import os
import neptune
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score

import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, ExponentialLR

from model import ViT

def get_gradient_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm

def train_model(model : ViT, optimizer, criterion, train_dl, test_dl, num_epochs, start_epoch=0):
    
    run = neptune.init_run(
        api_token=os.environ.get("NEPTUNE_API_TOKEN"),
        project='bng215/Model-Collapse'
    )
    
    scheduler = ExponentialLR(optimizer, 0.9)
    
    for epoch in range(start_epoch, num_epochs):
        
        model.train()
        with tqdm(train_dl) as tepoch:
            tepoch.set_description(f"Epoch: {epoch + 1}: Train stage")
            for batch in tepoch:
                
                img = batch['img'].cuda()
                labels = batch['label'].cuda()
                
                logits = model(img)
                
                loss = criterion(logits, labels)
            
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 15.0)
                optimizer.step()
                
                gnorm = get_gradient_norm(model)
                
                run['Loss (Train)'].append(loss.item())
                run['Gradient norm'].append(gnorm)
                run['Learning rate'].append(optimizer.param_groups[0]['lr'])

        
        model.eval()
        
        f1 = []
        loss_test = []
        
        with tqdm(test_dl) as tepoch:
            tepoch.set_description(f"Epoch: {epoch + 1}: Val stage")
            with torch.no_grad():
                for batch in tepoch:
                    img = batch['img'].cuda()
                    labels = batch['label'].cuda()
                    
                    logits = model(img)
                    
                    prob = F.softmax(logits, dim=-1)
                    
                    loss = criterion(logits, labels)
                    
                    loss_test.append(loss.item())
                    
                    f1.append(f1_score(labels.cpu(), prob.argmax(-1).cpu(), average='macro'))
                
        run['F1 (Test)'].append(np.mean(f1))
        run['Loss (Test)'].append(np.mean(loss_test))
        
        scheduler.step()
         
                    
        torch.save(model.state_dict(), f'weights/ep_{epoch+1}.pth')
                    
                    
                    
            
                
                