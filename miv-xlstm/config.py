def load_config():
    """
    Modify this config in order to test different default hyperparameters
    """
    config = {
        "file_path": "PATH/TO/cleaned_data.csv",
        "target": "vancomycin_administration",
        "pad_value": -999,
        "batch_size": 512,
        "output_size": 1,
        "context_length": 14,
        "hidden_size": 32,
        "num_blocks": 1,
        "num_epochs": 8,
        "learning_rate": 0.00239,
        "loss_weight": 4.86,
        "num_heads": 4,
        "k_folds": 5,
        "dropout": 0.3,
        "proj_factor": 1,
        "conv1d": 4,
        "project": "YOUR_WANDB_PROJECT",
        "how_much_data_used": 0.1,
    }
    return config