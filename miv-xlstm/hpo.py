import os
import time

import torch
import torch.nn as nn
import wandb
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from data_augmentation import load_and_preprocess_data
from models import xLSTMModel
from training import train_model
from helpers import load_config, WandbLogger

def hpo_trainable(config):
    """
    config: Ray overwrites the config received in the arguments at runtime
    """
    # def config is the one we set beforehand (not for hpo)
    # run config is managed by ray tune this way making hpo possible
    def_config = load_config()
    logger = WandbLogger(config=config, project=def_config["project"], on_off="online") 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, final_val_loader, num_features = load_and_preprocess_data( # when using the final hold-out validation we also validated using k_fold so it only used in that file, but it is here as well if anyone reusing this code needs it
        file_path = def_config["file_path"], 
        target = def_config["target"],
        pad_value = def_config["pad_value"],
        time_steps = def_config["context_length"],
        batch_size = def_config["batch_size"],
        val_cutoff=0.9,
    )
    print("number of features:", num_features)
    print("hyperparameter optimization loop")
    model = xLSTMModel(
        input_size = num_features, 
        hidden_size = config["hidden_size"], 
        num_blocks = config["num_blocks"], 
        output_size = def_config["output_size"], 
        context_length = def_config["context_length"]-1, # as only 13 days of data is used for train
        num_heads = config["num_heads"], 
        dropout = config["dropout"], 
        proj_factor = config["proj_factor"],
        ).to(device)

    # penalize the model if not detecting target patient
    class_weights = torch.tensor([config["loss_weight"]]).to(device) # this is correlated to batch_size?
    # loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])

    # train the model
    t0 = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, device, def_config["num_epochs"])
    time_taken = time.time() - t0
    wandb.log({'training_time_seconds': time_taken})
    wandb.finish()
    print("Training finished.")

def hpo_pipeline():
    """
    Usage: Modify the hyperparameter search space to your needs here

    Note: If you want to use more advanced schedulers like PBT, you'll have to implement checkpointing
    """
    search_space = {
        # hyperparameter search space
        "learning_rate": tune.qloguniform(1e-5, 5e-4, 5e-6), # uniformly between 0.0001 and 0.001 in log space with multiplies  of 0.00005
        "hidden_size": tune.choice([256, 512]), 
        "num_blocks": tune.grid_search([1, 2]),
        "loss_weight": tune.uniform(2, 64), # this is based on the specific target
        "num_heads": tune.grid_search([4,8,16]),
        "dropout": tune.choice([0.1, 0.2, 0.3]), 
        "proj_factor": tune.choice([1,1.4,2,3,4,5]) # value used by the creators of the original xlstm repository: 1.4
    }
    tuner = tune.Tuner(
        tune.with_resources(hpo_trainable, {"gpu": 1, "cpu": 2}),
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=ASHAScheduler(metric="auprc", mode="max"),
        ),
        param_space=search_space,
    )
    results= tuner.fit()
    print(results.get_best_result(metric="auprc", mode="max").config)