# miv-xlstm

# Quickstart

All tests were run on Python 3.10.12
In case of CUDA usage, we tested using Cuda compilation tools, release 12.6, v12.6.68

Create a python virtual environment and active it: 
```
python -m venv venv
source venv/bin/activate
```
Install dependencies inside virtual environment: 
```
pip install -r requirements.txt
```

For the data cleaning, change the path parameters at the top of data_cleaning.py, and then run it afterwards:
```
python data_cleaning.py
```

For hyperparameter optimization or k-fold cross-validation, change the config in `config.py` to your desired parameters. Don't forget to change the path to the dataset and modify the wandb project.

In `miv_xlstm.py` change to your desired mode either `k_fold` or `hpo`.
Afterwards, to run the pipeline:
```
python miv_xlstm.py
```

# Known issues

When using numpy > 2.0 version the scripts break. The data cleaning processes were using `numpy==1.26.4` while during experimentation and model training for external reasons we had to roll back to `numpy==1.24.4`.

# Ethics statement

The data for this study was derived from MIMIC-IV, a comprehensive critical care database of patients admitted to the Beth Israel Deaconess Medical Center (BIDMC) in Boston, Massachusetts. The collection and de-identification processes for MIMIC-IV received full approval from both BIDMC and MIT institutional review boards (IRBs). According to the publication, patient consent requirements were waived since the research had no impact on clinical care and all data were fully de-identified. Given that MIMIC-IV is a publicly available research database, additional local IRB approval was not required for this research. All subsequent data handling, including pre-processing, analysis, and machine learning applications were conducted in adherence to the established MIMIC-IV usage guidelines and regulations.

# References

[miv-xlstm paper](https://csutora.com/miv-xlstm) <br>
[MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/3.0/)
