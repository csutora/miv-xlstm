from config import load_config
from k_fold import k_fold_pipeline
from hpo import hpo_pipeline


def main(mode="kfold"):
    config = load_config()    
    if mode=="hpo":
        hpo_pipeline()
    elif mode=="kfold":
        k_fold_pipeline(config)
    

if __name__ == "__main__":
    main(mode="hpo")