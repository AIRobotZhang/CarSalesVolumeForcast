# -*- coding: utf-8 -*-

import time
from utils import dataset
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
from models.LSTM import LSTM
from sklearn import metrics
from torch.utils.data import DataLoader
from copy import deepcopy
import math
from pandas import DataFrame
import pandas as pd

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def get_score(prediction, target, model2index):
    prediction = prediction.reshape(-1)
    print(prediction)
    print(target)
    N = len(model2index.keys())
    score = 0
    for key in model2index:
        index_list = model2index[key]
        model_sum = 0.0
        true_sum = 0.0
        for ii in index_list:
            # print(model_sum)
            model_sum += (prediction[ii]-target[ii])*(prediction[ii]-target[ii])
            true_sum += target[ii]
        model_sum = math.sqrt(model_sum/len(index_list))
        score += model_sum/(true_sum/len(index_list))
    score = 1-score/N

    return score


def train_model(model, X_train, y_train, y_mean, X_val, y_val, epoch, model2index, lr, step):
    PATH = 'results/'+str(step)+'.prediction.pytorch.model.pkl'
    max_score = float('-inf')
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = torch.nn.MSELoss()
    # X_train = torch.from_numpy(X_train.astype(np.float32))
    # y_train = torch.from_numpy(y_train.astype(np.float32))
    # X_val = torch.from_numpy(X_val.astype(np.float32))
    # y_val = torch.from_numpy(y_val.astype(np.float32))

    # train_y = y_train[:,step].view(-1,1)
    # # print(y_train.mean())
    # y_mean = train_y.mean()
    # valid_y = y_val[:,step]

    for i in range(epoch):
        model.train()
        optim.zero_grad()

        prediction = model(X_train)
        loss = loss_fn(prediction, y_train-y_mean)

        loss.backward()
        # clip_gradient(model, 1e-1)
        optim.step()
        
        if i % 10 == 0:
            model.eval()
            val_prediction = model(X_val)
            # print((valid_y-y_mean).numpy())
            score = get_score(np.expm1(val_prediction.detach().numpy()+y_mean.numpy()), np.expm1(y_val.numpy()), model2index)
            if score > max_score:
                max_score = score
                torch.save(model.state_dict(), PATH)
            logger.info('%d: Epoch:%d, Training Loss:%.4f, Valid Score:%.4f', step, i, loss.item(), score)


def main():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument("--epoch", default=100, type=int,
                        help="the number of epoches needed to train")
    parser.add_argument("--lr", default=2e-5, type=float,
                        help="the learning rate")
    parser.add_argument("--classifier", default='lstm', type=str,
                        help="the classifier, such as LSTM, CNN ...")
    parser.add_argument("--hidden_size", default=64, type=int,
                        help="the hidden size")
    parser.add_argument("--output_size", default=1, type=int,
                        help="the output label size")
    parser.add_argument("--early_stopping", default=15, type=int,
                        help="Tolerance for early stopping (# of epochs).")
    parser.add_argument("--load_model", default=None, type=str,
                        help="load pretrained model for testing")

    parser.add_argument('--n_models', nargs='+')

    args = parser.parse_args()

    path  = 'D:/MyDocument/Project/CarSalesPrediction/'
    T = 12
    data = dataset.read_data(path)
    df, feature_list, y_list = dataset.get_feature(data, T)
    train_x, train_y, valid_x, valid_y, test_x, model2index = dataset.get_Xy(df, T, feature_list, y_list)

    X_train, X_val, X_test = dataset.normalization(train_x, valid_x, test_x)
    input_size = X_train.shape[2]
    print("feature_dim: ", input_size)

    model = LSTM(args.output_size, [args.hidden_size], input_size)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(train_y.astype(np.float32))
    X_val = torch.from_numpy(X_val.astype(np.float32))
    y_val = torch.from_numpy(valid_y.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    
    y_mean = []
    train_y = []
    valid_y = []
    for step in range(4):
        yy = y_train[:,step].view(-1,1)
        train_y.append(yy)
        mean_y = yy.mean()
        y_mean.append(mean_y)
        valid_y.append(y_val[:,step])

    # print (X_val)
    # print(X_val)

    if args.load_model:
        evaluation_public = pd.read_csv(path+'test2_dataset/evaluation_public.csv')[['id','regMonth','forecastVolum']]
        id_list = evaluation_public['id'].values
        evaluation_result = DataFrame({'id':id_list})
        # forecastVolum = []
        for ii, model_name in enumerate(args.n_models):
            print(model_name)
            model.load_state_dict(torch.load(args.load_model+'/'+model_name))
            model.eval()
            val_prediction = model(X_val)
            score = get_score(np.expm1(val_prediction.detach().numpy()+y_mean[ii].numpy()), np.expm1(valid_y[ii].numpy()), model2index)
            logger.info('Valid Score:%.4f', score)
            test_prediction = model(X_test)
            test_prediction = np.expm1(test_prediction.detach().numpy()+y_mean[ii].numpy()).reshape(-1).tolist()
            evaluation_public.loc[(evaluation_public.regMonth==ii+1), 'forecastVolum'] = test_prediction
            # forecastVolum.extend(test_prediction)
            del model
            model = LSTM(args.output_size, [args.hidden_size], input_size)
        # evaluation_result['forecastVolum'] = forecastVolum
        evaluation_public[['id','forecastVolum']].round().astype(int).to_csv('evaluation_public.csv', index=False)
        exit()

    for i in range(4):
        train_model(model, X_train, train_y[i], y_mean[i], X_val, valid_y[i], args.epoch, model2index, args.lr, i)


if __name__ == "__main__":
    main()
