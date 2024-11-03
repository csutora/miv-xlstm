import os
import time

import torch
import torch.nn as nn
import wandb

from data_augmentation import load_and_preprocess_data_k_fold
from models import xLSTMModel
from training import train_model
from validation import validate
from helpers import WandbLogger

def k_fold_pipeline(config):
    """
    Using the config retrieved from the hpo
    usage
        config = load_json_config()
        k_fold_pipeline(config)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_loaders, final_val_loader, num_features = load_and_preprocess_data_k_fold(
        file_path=config["file_path"], 
        target=config["target"],
        pad_value = config["pad_value"],
        time_steps=config["context_length"],
        batch_size=config["batch_size"],
        k_folds=config["k_folds"],
        val_cutoff=0.9,
        scale_percentile=config["how_much_data_used"],
    
    )
    print("number of features:", num_features)
    print("training via k-fold cross-validation")
    for i, (train_loader, val_loader) in enumerate(fold_loaders):
        logger = WandbLogger(config=config, project=config["project"], on_off="online") 
        model = xLSTMModel(
            input_size = num_features, 
            hidden_size = config["hidden_size"], 
            num_blocks = config["num_blocks"], 
            output_size = config["output_size"], 
            context_length = config["context_length"]-1,
            num_heads = config["num_heads"], 
            dropout = config["dropout"], 
            proj_factor = config["proj_factor"],
            ).to(device)

        # penalize the model if not detecting target patient
        class_weights = torch.tensor([config["loss_weight"]]).to(device)
        # loss and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])

        # train the model and measure time
        t0 = time.time()
        count_kfold = config["k_folds"]
        print(f"({i}/{count_kfold}) training...")
        train_model(model, train_loader, val_loader, criterion, optimizer, device, config["num_epochs"])
        time_taken = time.time() - t0

        # test using hold-out validation data
        test_loss, test_auroc, test_auprc = validate(model, final_val_loader, criterion, device)
        test_kpis = {'test_loss': test_loss, 'test_auroc': test_auroc, "test_auprc": test_auprc}
        wandb.log(test_kpis)

        print(f"({i}/{count_kfold}) training time: {time_taken} seconds")
        wandb.log({'training_time_seconds': time_taken})
        wandb.finish()
    print("Training finished.")