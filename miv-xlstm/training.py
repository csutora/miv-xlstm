import torch
from torchmetrics.classification import BinaryF1Score
import tqdm
import wandb
from ray import train

from validation import validate

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Trains model for X amount of epoch while logging training f1-score and loss to wandb
    Calls and logs validation after each epoch
    """
    f1_score = BinaryF1Score().to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", miniters=int(len(train_loader)/2))
        
        for i, (inputs, targets) in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # we are only using the last day to calculate loss on
            actual_targets = targets[:,-1].reshape((-1, 1)).to(device)
            loss = criterion(outputs, actual_targets).to(device)
            loss.backward()
            optimizer.step()

            # KPI calculation
            detached_out = outputs.detach().squeeze()
            train_f1_score = f1_score(detached_out, targets[:,-1])
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1), 'f1_score': train_f1_score})
            wandb.log({'loss': running_loss / (i + 1),'epoch': epoch, 'train_f1_score': train_f1_score})

        # validation loop    
        val_loss, binary_auroc, binary_auprc = validate(model, val_loader, criterion, device)
        kpis = {'validation loss': val_loss, 'validation_auroc': binary_auroc, 'validation_auprc': binary_auprc}
        progress_bar.set_postfix(kpis)
        print(kpis)
        wandb.log(kpis)
        # validation auprc will be the one being maximized
        train.report({'auprc': binary_auprc})
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")