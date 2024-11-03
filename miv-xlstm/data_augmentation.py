import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler



def load_and_preprocess_data(file_path, target='myocardial_infarction', pad_value = 0,
                             time_steps=14, batch_size=32, tt_split=0.67, val_cutoff=0.9):
    """
    Runs data augmentation pipeline and creates DataLoaders for hyperparameter optimization and hold-out validation
    """
    
    X_tensor, y_tensor = load_and_create_targets_and_pad(file_path, target, pad_value, time_steps)
    train_loader, val_loader, final_val_loader, feature_count = create_loaders(X_tensor, y_tensor, pad_value, time_steps, batch_size, tt_split, val_cutoff)
    
    return train_loader, val_loader, final_val_loader, feature_count


def load_and_preprocess_data_k_fold(file_path, target='myocardial_infarction', pad_value = 0,
                                              time_steps=14, batch_size=32, k_folds=5, val_cutoff=0.9, scale_percentile = 0.1):
    """
    Runs data augmentation pipeline and creates K-fold and hold-out validation DataLoaders
    """
    X_tensor, y_tensor = load_and_create_targets_and_pad(file_path, target, pad_value, time_steps)
    fold_loaders, final_val_loader, feature_count = create_loaders_k_fold(X_tensor, y_tensor, pad_value, time_steps, batch_size, k_folds, val_cutoff, scale_percentile)
    
    return fold_loaders, final_val_loader, feature_count


def load_and_create_targets_and_pad(file_path, target, pad_value, time_steps):
    """
    Creates targets based on different conditions
    Scales the data using StandardScaler
    Pads data to 14 day sequences
    """
    # load the data, handling the index properly
    df = pd.read_csv(file_path, index_col='HADMID_DAY')

    # define the target variable
    if target == 'myocardial_infarction':
        df[target] = (((df['troponin-t'].astype(float) > 0.4) & (df['ckd'].astype(int) == 0))).astype(int)
        toss = ['troponin-t', 'troponin-t_iqr', 'troponin-t_min', 'troponin-t_max', 'ckd', 'myocardial_infarction']
        features = df.drop(columns=toss)

    elif target == 'sepsis': 
        
        # based on SIRS criteria, >= 2 meets SIRS definition
        # and SOFA score, >= 2 meets SOFA definition, as recommended by Sepsis-3
        # only partials as we don't have all the features

        df['sirs_temperature'] = df['temperature_fahrenheit'].astype(float).apply(lambda x: 1 if x > 100.4 or (x < 96.8 and x != 0) else 0)
        df['sirs_heart_rate'] = (df['heart_rate'].astype(float) > 90).astype(int)
        df['sirs_respiratory_rate'] = (df['respiratory_rate'].astype(float) > 20).astype(int)
        df['sirs_wbcs_or_bands'] = (df['white_blood_cell'].astype(float).apply(lambda x: 1 if x > 12 or (x < 4 and x != 0) else 0)
                                     | (df['bands'].astype(float) > 10).astype(int)
                                    ).astype(int)
        df['sirs_points'] = (df['sirs_temperature']
                             + df['sirs_heart_rate']
                             + df['sirs_respiratory_rate']
                             + df['sirs_wbcs_or_bands']
                            ).astype(int)
        
        df['sofa_platelets'] = (df['platelet_count'].astype(float) < 150).astype(int)
        df['sofa_artbp'] = (
            np.minimum(
                (df['arterial_blood_pressure_systolic'].astype(float) + df['arterial_blood_pressure_diastolic'].astype(float)) / 2,
                (df['art_bp_systolic'].astype(float) + df['art_bp_diastolic'].astype(float)) / 2
            ) < 70
        ).astype(int)
        df['sofa_creatinine'] = (df['creatinine'].astype(float) > 1.2).astype(int)
        df['sofa_points'] = (df['sofa_platelets']
                             + df['sofa_artbp']
                             + df['sofa_creatinine']
                            ).astype(int)

        df[target] = (((df['sirs_points'].astype(int) >= 2) | (df['sofa_points'].astype(int) >= 2)) & (df['suspected_infection'].astype(int) == 1)).astype(int)
        toss = ['sirs_temperature', 'sirs_heart_rate', 'sirs_respiratory_rate', 'sirs_wbcs_or_bands', 'sirs_points', 'sofa_platelets', 'sofa_artbp', 'sofa_creatinine', 'sofa_points', 'suspected_infection', 'sepsis'] # possibly drop all inputs but they're non-linearly correlated so keeping them for now
        features = df.drop(columns=toss)
    
    elif target == 'vancomycin_administration':
        df[target] = (df['vancomycin'].astype(float) > 0).astype(int)
        toss = ['vancomycin', 'vancomycin_administration']
        features = df.drop(columns=toss)

    else: raise ValueError('invalid target variable provided')

    # normalize the features
    scaler = StandardScaler()
    normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=df.index)
    normalized_features[target] = df[target]
    print(f"value counts after normalization: {normalized_features[target].value_counts()} ")
    
    # add padding
    padded = pad_sequences(normalized_features, target, time_steps, lb=3, pad_value=pad_value)
    padded_cleaned = padded.drop(columns= ['HADM_ID', 'DAY'])
    print(f"value counts after padding: {padded_cleaned[target].value_counts()} ")
    padded_cleaned_wo_target = padded_cleaned.drop(columns=[target])

    # prepare X and y
    X_tensor = padded_cleaned_wo_target.values.astype('float32')
    y_tensor = padded_cleaned[target].values.astype('float32')

    return X_tensor, y_tensor


def create_loaders(X_tensor, y_tensor, pad_value, time_steps, batch_size, tt_split, val_cutoff):
    """
    Creates a normal training and validation loader while cutting the last 10% off for final testing
    This is used during hyperparameter optimization
    """
    # split 10% off so we can use untouched data for the final validation
    X, X_val_final = train_test_split(X_tensor, lookback=time_steps, percentage=val_cutoff)
    y, y_val_final = train_test_split(y_tensor, lookback=time_steps, percentage=val_cutoff)

    # transform and create dataloader for that final 10%
    X_val_final_transf, y_val_final_transf = transform_dataset(X_val_final, y_val_final, pad_value = pad_value, lookback=time_steps, lb=3)
    final_val_loader = DataLoader(TensorDataset(X_val_final_transf, y_val_final_transf), batch_size=batch_size, drop_last=True)
    print(f"shapes after transforming final validation with lookback: x_val: {X_val_final_transf.shape}, y_val: {y_val_final_transf.shape}")

    # split the data into train and validation
    X_train, X_val = train_test_split(X, lookback=time_steps, percentage=tt_split)
    y_train, y_val = train_test_split(y, lookback=time_steps, percentage=tt_split)
    print(f"after train test split: x_train shape:{X_train.shape},y_train shape: {y_train.shape}")
    print(f"after train test split: x_val shape:{X_val.shape},y_val shape: {y_val.shape}")

    X_train_transf, y_train_tranfs = transform_dataset(X_train, y_train, pad_value = pad_value, lookback=time_steps, lb=3)
    X_val_transf, y_val_tranfs = transform_dataset(X_val, y_val, pad_value = pad_value, lookback=time_steps, lb=3)

    print(f"after transforming with lookback: x_train shape:{X_train_transf.shape},y_train shape: {y_train_tranfs.shape}")
    print(f"after transforming with lookback: x_val shape:{X_val_transf.shape},y_val shape: {y_val_tranfs.shape}")

    unique_y_train, count_y_train = y_train_tranfs.unique(return_counts=True)
    unique_y_val, count_y_val = y_val_tranfs.unique(return_counts=True)
    print(f"y_train unique: {unique_y_train}, count: {count_y_train}")
    print(f"y_val unique: {unique_y_val}, count: {count_y_val}")

    train_loader = DataLoader(TensorDataset(X_train_transf, y_train_tranfs), batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val_transf, y_val_tranfs),batch_size=batch_size, drop_last=True)

    return train_loader, val_loader, final_val_loader, X_tensor.shape[-1]



def create_loaders_k_fold(X_tensor, y_tensor, pad_value, time_steps, batch_size, k_folds, val_cutoff, scale_percentile):
    # split 10% off so we can use untouched data for the final validation
    X_100, X_val_final = train_test_split(X_tensor, lookback=time_steps, percentage=val_cutoff)
    y_100, y_val_final = train_test_split(y_tensor, lookback=time_steps, percentage=val_cutoff)
    print(f"shapes after splitting the final validation off: x_train: {X_100.shape}, y_train: {y_100.shape}")
    print(f"shapes of final validation: x_val: {X_val_final.shape}, y_val: {y_val_final.shape}")
    
    # percentile cutoff for data scaling law investigation
    X, X_unused = train_test_split(X_100, lookback=time_steps, percentage=scale_percentile)
    y, y_unused = train_test_split(y_tensor, lookback=time_steps, percentage=scale_percentile)
    print(f"shapes after only using {scale_percentile*100} % of train: x_train: {X.shape}, y_train: {y.shape}")


def create_loaders_k_fold(X_tensor, y_tensor, pad_value, time_steps, batch_size, k_folds, val_cutoff, scale_percentile):
    """
    Creates K amount of dataloaders, while keeps the last 10% of data for testing

    Returns:- List of tuples K amount of DataLoaders (training, validation), 
            - Final testing DataLoader
            - Feature count
    """
   # split 10% off so we can use untouched data for the final validation
    X_100, X_val_final = train_test_split(X_tensor, lookback=time_steps, percentage=val_cutoff)
    y_100, y_val_final = train_test_split(y_tensor, lookback=time_steps, percentage=val_cutoff)
    print(f"shapes after splitting the final validation off: x_train: {X_100.shape}, y_train: {y_100.shape}")
    print(f"shapes of final validation: x_val: {X_val_final.shape}, y_val: {y_val_final.shape}")
    
    # percentile cutoff for data scaling law investigation
    X, X_unused = train_test_split(X_100, lookback=time_steps, percentage=scale_percentile)
    y, y_unused = train_test_split(y_tensor, lookback=time_steps, percentage=scale_percentile)
    print(f"shapes after only using {scale_percentile*100} % of train: x_train: {X.shape}, y_train: {y.shape}")

    # transform and create dataloader for that final 10%
    X_val_final_transf, y_val_final_transf = transform_dataset(X_val_final, y_val_final, pad_value = pad_value, lookback=time_steps, lb=3)
    final_val_loader = DataLoader(TensorDataset(X_val_final_transf, y_val_final_transf), batch_size=batch_size, drop_last=True)
    print(f"shapes after transforming final validation with lookback: x_val: {X_val_final_transf.shape}, y_val: {y_val_final_transf.shape}")

    # k-fold cross-validation
    # we have a unique data situation where we can only divide every 14th days,
    # so we had do write a custom k-fold :)
    n, k, d = X.shape[0], k_folds, time_steps # number of samples, number of folds, number of days in a window
    # calculate the size each part would be if we didn't have the constraint of division by 14
    ideal_size = n / k

    partitions = []
    start = 0
    for i in range(k):
        # calculate the end of this part
        end = min(n, round((i + 1) * ideal_size))
        # adjust end to be a multiple of d
        end = min(n, ((end + d - 1) // d) * d - 1)
        # if it's the last part, make sure it includes the last element
        if i == k - 1:
            end = n - 1
        partitions.append((start, end))
        start = end + 1

    print(f"k-fold cv partitions: {partitions}")

    fold_loaders = []
    i = 1
    for start, end in partitions:
        X_train, X_val = np.concatenate([X[:start], X[end+1:]]), X[start:end+1]
        y_train, y_val = np.concatenate([y[:start], y[end+1:]]), y[start:end+1]

        X_train_transf, y_train_transf = transform_dataset(X_train, y_train, pad_value=pad_value, lookback=time_steps, lb=3)
        X_val_transf, y_val_transf = transform_dataset(X_val, y_val, pad_value=pad_value, lookback=time_steps, lb=3)

        train_loader = DataLoader(TensorDataset(X_train_transf, y_train_transf), batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(TensorDataset(X_val_transf, y_val_transf), batch_size=batch_size, drop_last=True)

        print(f"fold {i}/{k_folds}:")
        print(f"- x_train: {X_train_transf.shape}, y_train: {y_train_transf.shape}")
        print(f"- x_val: {X_val_transf.shape}, y_val:   {y_val_transf.shape}")
        i += 1

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders, final_val_loader, X_tensor.shape[-1]   

def train_test_split(timeseries, lookback = 14, percentage = 0.67):
    """
    Partitions the data according the the preset percentage
    """
    # train-test split for time series
    train_size = int((len(timeseries)/lookback)*percentage)*lookback
    train, test = timeseries[:train_size], timeseries[train_size:]
    return train, test

def transform_dataset(x,y, pad_value = 0, lookback=14, lb = 3):
    """
    Transform a time series into a prediction dataset patient wise
    Patients are ordered by hadmid_day into lookback (most likely 14) day blocks
    This creates lookback-1 (13) day windows where we want to predict the n+1 day (thus the lookback-1)
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction, defaults to 14
        lb: lower bound defines what is the minimal window size when creating patient windows
    """
    X_data, y_data = [], []
    distribution_of_y_true = torch.zeros(lookback-1)
    padding_window_feature = (pad_value * np.ones((lookback-1-(lb-1), x.shape[-1])))
    padding_window_target = np.zeros((lookback-1-(lb-1)))
    #for i in range(len(x)-lookback): # like a stock timeseries, treating everyone as one entity
    for i in range(0, len(x), lookback): # patient wise
        window = lookback - 1 # window to use for learning stuff
        feature = x[i:i+window]
        target = y[i+1:i+window+1]
        padded_feature = np.concatenate([padding_window_feature, feature], axis=0).astype(np.float32)
        padded_target = np.concatenate([padding_window_target, target], axis=0).astype(np.float32)
        windowed_feature = []
        windowed_target = []
        for i in range (0, padding_window_feature.shape[0]):
            windowed_feature = padded_feature[i:i+window]
            windowed_target = padded_target[i:i+window]
            if not all(x == 0 for x in windowed_feature[-(lb-1)]):
                X_data.append(windowed_feature)
                y_data.append(windowed_target)
        distribution_of_y_true += target
        X_data.append(feature)
        y_data.append(target)
    return torch.tensor(X_data), torch.tensor(y_data)


def pad_sequences(df, target, time_steps = 14,lb=3, pad_value=0):
    ''' 
    Takes a DataFrame to operate on. lb is a lower bound to discard 
    time_steps is the number of time steps to pad or truncate to. All entries are padded to time_steps
    and the padding is done before the data

    Consider lower bound, using 3 corresponds to 400k rows.  (2: 700k; 1:1M)
    Consider pad value: choices -1, -999 or 0
    '''

    # split the HADM_ID_DAY column into separate HADM_ID and DAY columns
    df['HADM_ID'] = df.index.str.split('_').str[0]
    df['DAY'] = df.index.str.split('_').str[1]

    # keep all the positives even if it's under the lower bound
    def keep_group(group):
        return (len(group) > lb)

    # filter groups, keeping those that meet the criteria
    df = df.groupby('HADM_ID').filter(keep_group).reset_index(drop=True)

    # create a DataFrame with pad_value for padding
    pad_df = pd.DataFrame(pad_value * np.ones((time_steps, len(df.columns))), columns=df.columns)
    pad_df.loc[:,target] = np.zeros(len(pad_df))
    def pad_group(group):
        if len(group) >= time_steps:
            # if the group is already at or exceeding time_steps, return the last time_steps rows
            return group.iloc[-time_steps:]
        else:
            # pad the beginning and then add the real data
            padding_needed = time_steps - len(group)
            padded_part = pad_df.iloc[:padding_needed]
            return pd.concat([padded_part, group], axis=0)

    # apply padding AFTER
    df = df.groupby('HADM_ID').apply(pad_group).reset_index(drop=True)
    return df