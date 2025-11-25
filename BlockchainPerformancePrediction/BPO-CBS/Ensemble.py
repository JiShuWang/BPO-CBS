import os
import pickle
from argparse import ArgumentParser

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

seed = 42
torch.manual_seed(seed)
torch.set_float32_matmul_precision('high')


# batch the training dataset
# prepare dataset
class BlockChainDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data).type(torch.float32)
        self.y_data = torch.from_numpy(label).type(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3, GradientBoostingRegressor=None, LightGBMRegressor=None, RandomForestRegressor=None, ExtraTreesRegressor=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.backbone = backbone
        self.criterion = nn.MSELoss()

        self.RandomForestRegressor = RandomForestRegressor
        self.GradientBoostingRegressor = GradientBoostingRegressor
        self.LightGBMRegressor = LightGBMRegressor
        self.ExtraTreesRegressor = ExtraTreesRegressor

    def forward(self, x):
        # embedding = self.GradientBoosting(x)
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = torch.cat((y_hat, torch.reshape(torch.from_numpy(self.GradientBoostingRegressor.predict(x)).cuda(), (-1, 1))), dim=1)
        y_hat = torch.cat((y_hat, torch.reshape(torch.from_numpy(self.LightGBMRegressor.predict(x)).cuda(), (-1, 1))), dim=1)
        y_hat = torch.cat((y_hat, torch.reshape(torch.from_numpy(self.RandomForestRegressor.predict(x)).cuda(), (-1, 1))), dim=1)
        y_hat = torch.cat((y_hat, torch.reshape(torch.from_numpy(self.ExtraTreesRegressor.predict(x)).cuda(), (-1, 1))), dim=1)
        y_hat = torch.reshape(torch.mean(y_hat, dim=1), (-1, 1))
        loss = self.criterion(y_hat, y)
        MAE = metrics.mean_absolute_error(y, y_hat)
        RMSE = metrics.mean_squared_error(y, y_hat) ** 0.5
        MAPE = metrics.mean_absolute_percentage_error(y, y_hat)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_MAE", MAE, prog_bar=True)
            self.log(f"{stage}_RMSE", RMSE, prog_bar=True)
            self.log(f"{stage}_MAPE", MAPE, prog_bar=True)
        return MAE, RMSE, MAPE

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        MAE, RMSE, MAPE = self.evaluate(batch, "test")
        print('MAE:{MAE} RMSE:{RMSE} MAPE:{MAPE}'.format(MAE=MAE, RMSE=RMSE, MAPE=MAPE))
        return MAE, RMSE, MAPE

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def cli_main():
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dataset', type=str, default="MMBPD_Distributed", help='Indicates which data set to use')
    parser.add_argument('--task', type=str, default="Throughput", help='Indicates which task to perform')
    # parser.add_argument('--task', type=str, default="Latency", help='Indicates which task to perform')
    args = parser.parse_args()
    # ------------
    # data
    # ------------
    raw_data = pd.read_csv("../data/" + str(args.dataset) + '.csv').values
    if args.dataset == "BPD-1":
        X = raw_data[:, :2]
        if args.task == "Latency":
            Y1 = raw_data[:, 2].reshape((-1, 1))
        else:
            Y1 = raw_data[:, 3].reshape((-1, 1))
    elif args.dataset == "HFBTP":
        X = raw_data[:, :3]
        if args.task == "Latency":
            Y1 = raw_data[:, 4].reshape((-1, 1))
        else:
            Y1 = raw_data[:, 3].reshape((-1, 1))
    elif args.dataset == "MMBPD_Distributed" or "MMBPD_Collective":
        X = raw_data[:, 1:3]
        if args.task == "Latency":
            Y1 = raw_data[:, 4].reshape((-1, 1))
        else:
            Y1 = raw_data[:, -2].reshape((-1, 1))
    results = []
    KF = KFold(n_splits=5, random_state=seed, shuffle=True)
    i = 1
    os.makedirs(os.path.join('Models', str(args.dataset), str(args.task)), exist_ok=True)
    os.makedirs(os.path.join('Results', str(args.dataset), str(args.task)), exist_ok=True)
    for train_index, test_index in KF.split(X):
        Xtrain1, Xtest1 = X[train_index], X[test_index]
        Ytrain1, Ytest1 = Y1[train_index], Y1[test_index]
        # property scaling
        min_max_scaler1 = preprocessing.MinMaxScaler()
        # Scaling training set data
        Xtrain1_minmax = min_max_scaler1.fit_transform(Xtrain1)
        # Apply the same scaling to the test set data
        Xtest1_minmax = min_max_scaler1.transform(Xtest1)

        train_dataset = BlockChainDataset(Xtrain1_minmax, Ytrain1)
        val_dataset = BlockChainDataset(Xtest1_minmax, Ytest1)
        test_dataset = BlockChainDataset(Xtest1_minmax, Ytest1)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=23, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=Ytest1.shape[0], shuffle=False, pin_memory=True, num_workers=23, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=Xtest1.shape[0], shuffle=False, pin_memory=True, num_workers=23, persistent_workers=True)

        GradientBoosting = GradientBoostingRegressor(n_estimators=500, random_state=42)
        GradientBoosting.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'GBR' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(GradientBoosting, file)

        LightGBM = lgb.LGBMRegressor(objective='regression', num_leaves=31, n_estimators=500, random_state=42)
        LightGBM.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'LightGBM' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(LightGBM, file)

        RandomForest = RandomForestRegressor(n_estimators=500, random_state=42)
        RandomForest.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'RFR' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(RandomForest, file)

        ExtraTrees = ExtraTreesRegressor(n_estimators=500, random_state=42)
        ExtraTrees.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'ETR' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(ExtraTrees, file)

        MLP = Backbone()
        model = LitClassifier(MLP, args.learning_rate, GradientBoostingRegressor=GradientBoosting, LightGBMRegressor=LightGBM, RandomForestRegressor=RandomForest,
                              ExtraTreesRegressor=ExtraTrees)
        MLP.cuda()
        model.cuda()

        early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.000, patience=10, verbose=True, mode="min")
        # ------------
        # training
        # ------------
        trainer = pl.Trainer(max_epochs=200, check_val_every_n_epoch=10,
                             callbacks=[early_stop_callback], accelerator="gpu")
        trainer.fit(model, train_loader, val_loader)

        # trainer.save_checkpoint(os.path.join('Models', str(args.dataset), str(args.task), 'Ensemble' + str(i) + '.ckpt'))
        torch.save(MLP.state_dict(), os.path.join('Models', str(args.dataset), str(args.task), 'Ensemble' + str(i) + '.ckpt'))
        # ------------
        # testing
        # ------------
        result = trainer.test(model, test_loader)
        result = list(result[0].values())[1:]
        i = i + 1
        results.append(result)

    os.makedirs(os.path.join(str(args.dataset), str(args.task)), exist_ok=True)
    results = np.array(results)

    resultpd = pd.DataFrame(np.vstack((np.vstack((results, results.mean(0).reshape((1, -1)))), results.std(0))))
    if args.task == "Latency":
        resultpd.columns = ['latency_MAE', 'latency_RMSE', 'latency_MAPE']
    if args.task == "Throughput":
        resultpd.columns = ['throughput_MAE', 'throughput_RMSE', 'throughput_MAPE']

    # save evaluation result
    resultpd.to_csv(os.path.join(str("Results"), str(args.dataset), str(args.task), 'Metrics.csv'), index=False)


if __name__ == '__main__':
    cli_main()

