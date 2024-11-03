import numpy as np
import torch
from torcheval.metrics import BinaryAUPRC, BinaryAUROC
from torchmetrics import PrecisionRecallCurve
from torchmetrics.classification import BinaryConfusionMatrix, BinaryROC, BinaryF1Score
import wandb

from helpers import make_wandb_plot

def validate(model, val_loader, criterion, device):
    """
    Validation loop logging our tracked KPIs to wandb
    """
    binary_auprc = BinaryAUPRC().to(device) # torcheval
    binary_auroc = BinaryAUROC().to(device) # torcheval
    roc = BinaryROC().to(device) # torchmetrics lightning
    conf_mat = BinaryConfusionMatrix().to(device) # torchmetrics lightning
    pr_curve = PrecisionRecallCurve(task='binary').to(device) # torchmetrics lightning
    f1_score = BinaryF1Score().to(device) # torchmetrics lightning
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            actual_targets = targets[:,-1].reshape((-1, 1)).to(device)
            loss = criterion(outputs, actual_targets).to(device)
            
            # KPI calc
            outputs_squeezed = outputs.detach().squeeze().to(device)
            roc.update(outputs_squeezed, targets[:,-1].to(torch.int).to(device))
            conf_mat.update(outputs_squeezed, targets[:,-1].to(torch.int).to(device))
            pr_curve.update(outputs_squeezed, targets[:,-1].to(torch.int).to(device))
            f1_score.update(outputs_squeezed, targets[:,-1].to(torch.int).to(device))
            binary_auprc.update(outputs_squeezed, targets[:,-1])
            binary_auroc.update(outputs_squeezed, targets[:,-1])

            val_loss += loss.item()
            
    fpr, tpr, fpr_tpr_thresholds = roc.compute()
    precision, recall, p_r_thresholds = pr_curve.compute()
    val_f1_value = f1_score.compute()
    fig_conf_mat, ax = conf_mat.plot()
    conf_mat.reset()
    roc.reset()
    f1_score.reset()
    pr_curve.reset()
    wandb.log({'conf_mat': wandb.Image(fig_conf_mat)})
    wandb.log({'val_f1_score': val_f1_value})
    wandb.log({"ROC": make_wandb_plot(np.interp(np.linspace(0,1,1001), tpr.cpu().numpy(), fpr.cpu().numpy()), np.linspace(0,1,1001), "FPR", "TPR", "ROC Curve")})
    wandb.log({"PRC": make_wandb_plot(np.interp(np.linspace(0,1,1001), precision.cpu().numpy(), recall.cpu().numpy()), np.linspace(0,1,1001), "Recall", "Precision", "Precision Recall Curve")})
    
    return val_loss / len(val_loader), binary_auroc.compute().item(), binary_auprc.compute().item()