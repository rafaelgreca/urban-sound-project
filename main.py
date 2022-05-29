import pandas as pd
import os
import librosa
import numpy as np
import warnings
import torch
import torch.nn as nn
import argparse
import logging

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mlp import MLP

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, help="learning rate in the training step")
    parser.add_argument("--n_mfcc", type=int, help="number of mfcc features")
    parser.add_argument("--test_size", type=float, help="percentage of the dataset used to validate the model")
    parser.add_argument("--dropout_rate", type=float, help="dropout rate")
    args = parser.parse_args()

    df_metadata = pd.read_csv(os.path.join(os.getcwd(), 'data/UrbanSound8K.csv'), sep=',')
    df_features = pd.DataFrame()

    for (file_name, fold, class_id) in tqdm(zip(df_metadata['slice_file_name'],
                                                df_metadata['fold'],
                                                df_metadata['classID'])):
        file_path = os.path.join(os.getcwd(), f'data/fold{fold}', file_name)
        
        ## loading audio file
        audio, sr = librosa.load(file_path,
                                 sr=8000,
                                 res_type='kaiser_fast')

        ## extracting features from the audio using MFCC
        mfcc_feature = librosa.feature.mfcc(y=audio,
                                            sr=sr,
                                            n_mfcc=args.n_mfcc)

        mfcc_feature = np.mean(mfcc_feature.T, axis=0)

        temp_df = pd.DataFrame({'features': [mfcc_feature],
                                'class': [class_id]})

        df_features = pd.concat([df_features, temp_df], axis=0)

    ## converting features and labels list to numpy array
    X = np.array(df_features['features'].tolist())
    y = np.array(df_features['class'].tolist())

    ## splitting the data
    X_train, X_validation, y_train, y_validation = train_test_split(X,
                                                                    y,
                                                                    test_size=args.test_size,
                                                                    random_state=args.seed)
    
    ## converting to torch's tensor
    X_train = torch.from_numpy(X_train).to(device)
    X_validation = torch.from_numpy(X_validation).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_validation = torch.from_numpy(y_validation).to(device)

    ## training the model
    model = MLP(seed=args.seed, input_dim=args.n_mfcc, dropout_rate=args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # forward
        outputs = model(X_train)
        l = loss(outputs, y_train)

        # backwards
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % 5 == 0:
            logging.info(f'epoch: {epoch+1} | loss: {l.item()}')
    
    ## validating the model
    model.eval()
    total_corrects = 0
    total_samples = 0

    with torch.no_grad():
        outputs = model(X_validation)

        # returns the value and the index
        _, predictions = torch.max(outputs, 1)

        total_samples += y_validation.shape[0]
        total_corrects += (predictions  == y_validation).sum().item()

    logging.info(f'model accuracy: {total_corrects/total_samples}')