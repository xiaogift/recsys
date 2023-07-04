#!/usr/bin/env python3
# ============================================================================== #
# Recommender System @ MovieLens_100k
# Powered by xiaolis@outlook.com 202307
# ============================================================================== #
from pandas import read_csv, notnull
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler

import os, torch, numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wide_features = ['user_id', 'item_id', 'release_year', 'zip_code']
deep_features = ['age', 'gender', 'occupation', 'timestamp']

# ============================================================================== #
class MovieLens:

    def __init__(self, root='./movielens_data'):
        self._download_movielens_data(root)
        self._load_movielens_data(os.path.join(root,'ml-100k/'))
        self._transform_to_features()
        self._split_train_val()

    def _download_movielens_data(self, data_dir):
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        data_url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        data_zip_file = os.path.join(data_dir, 'ml-100k.zip')
        data_extract_dir = os.path.join(data_dir, 'ml-100k')
        if not os.path.exists(data_zip_file):
            from urllib import request
            print('Downloading MovieLens dataset...')
            request.urlretrieve(data_url, data_zip_file)
        if not os.path.exists(data_extract_dir):
            from zipfile import ZipFile
            print('Extracting dataset...')
            with ZipFile(data_zip_file, 'r') as zip_ref: zip_ref.extractall(data_dir)

    def _load_movielens_data(self, data_dir):
        ratings = read_csv( data_dir+'u.data', 
                            sep = '\t', 
                            header = None, 
                            names = ['user_id', 'item_id', 'rating', 'timestamp'] )
        users = read_csv( data_dir+'u.user', 
                          sep = '|',
                          header = None,
                          names = ['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                          encoding = 'latin-1')
        movies = read_csv( data_dir+'u.item', 
                           sep = '|', 
                           header = None, 
                           encoding = 'latin-1')
        seq_len = movies[range(5,24)].sum(axis=1).max()
        def func(x):
            genre = list(np.where(x.iloc[5:].values)[0])
            genre.extend([0] * (seq_len - len(genre)))
            return genre
        movies = read_csv(data_dir+'u.item', sep='|', header=None, encoding='latin-1')
        movies['genre'] = movies.apply(func, axis=1)
        movies.drop(columns=[1, *range(3, 24)], inplace=True)
        movies.columns = ['item_id', 'release_year', 'genre']
        movies.drop('genre',axis=1, inplace=True) ###
        fill_na = movies['release_year'].mode()[0][-4:]
        movies['release_year'] = movies['release_year'].apply(lambda x: x[-4:] if notnull(x) else fill_na).astype(int)
        self.data = ratings.merge(users, on='user_id', how='left').merge(movies, on='item_id', how='left')

    def _transform_to_features(self):
        mms = MinMaxScaler()
        self.data['timestamp'] = mms.fit_transform(self.data['timestamp'].values.reshape(-1, 1))
        onehot_col = ['gender', 'occupation'] # genre
        ohe = OneHotEncoder(sparse=False)
        for i in onehot_col: self.data[i] = ohe.fit_transform(self.data[[i]])
        label_col = ['user_id', 'item_id', 'release_year','zip_code']
        le = LabelEncoder()
        for i in label_col: self.data[i] = le.fit_transform(self.data[i])
        standard_col = ['age','rating']
        ss = StandardScaler()
        for i in standard_col: self.data[i] = ss.fit_transform(self.data[i].values.reshape(-1, 1))

    def _split_train_val(self, val_test_ratio=0.2):
        train_idx, val_idx = [], []
        for user_id in self.data['user_id'].unique():
            df_tmp = self.data[self.data['user_id'] == user_id]
            cnt = df_tmp.shape[0]
            val_test_cnt = int(cnt * val_test_ratio)
            train_cnt = cnt - val_test_cnt
            idx = df_tmp.index.tolist()
            train_idx.extend(idx[:train_cnt])
            val_idx.extend(idx[train_cnt:])
        self.trn_data = self.data.iloc[train_idx].reset_index(drop=True)
        self.val_data = self.data.iloc[val_idx].reset_index(drop=True)

    def get_trn_val(self): return self.trn_data, self.val_data

class MovieLensDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, index):
        current_row = self.data.iloc[index]
        return { "wide_inputs": torch.tensor(current_row[wide_features].values),
                 "deep_inputs": torch.tensor(current_row[deep_features].values.astype(np.float32), dtype=torch.float),
                 "rating": torch.tensor(current_row['rating'], dtype=torch.float) }

def get_dataloader():
    trn_data, val_data = MovieLens().get_trn_val()
    trn_dataloader = DataLoader( MovieLensDataset(trn_data), batch_size = 1024, shuffle = True )
    val_dataloader = DataLoader( MovieLensDataset(val_data), batch_size = 1000, shuffle = False )

    print(f'*** Movielens training data {len(trn_data)} as \n {trn_data.head()}')
    print(f'*** Movielens validation data {len(val_data)} as \n {val_data.head()}')
    return trn_dataloader, val_dataloader

# ============================================================================== #
class WideAndDeep(torch.nn.Module):
    def __init__(self, wide_dim, deep_dim, hidden_dim, output_dim):
        super(WideAndDeep, self).__init__()
        self.wide_linear = torch.nn.Linear(wide_dim, output_dim)
        self.deep_linear1 = torch.nn.Linear(deep_dim, hidden_dim)
        self.deep_linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, wide_inputs, deep_inputs):
        wide_output = self.wide_linear(wide_inputs.float())
        deep_output = F.relu(self.deep_linear1(deep_inputs))
        deep_output = self.deep_linear2(deep_output)
        return wide_output + deep_output

# ============================================================================== #
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_data in dataloader:
        wide_inputs = batch_data["wide_inputs"].to(device)
        deep_inputs = batch_data["deep_inputs"].to(device)
        ratings = batch_data["rating"].to(device)
        optimizer.zero_grad()
        outputs = model(wide_inputs, deep_inputs)
        loss = criterion(outputs, ratings.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ============================================================================== #
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_data in dataloader:
            wide_inputs = batch_data["wide_inputs"].to(device)
            deep_inputs = batch_data["deep_inputs"].to(device)
            ratings = batch_data["rating"].to(device)
            outputs = model(wide_inputs, deep_inputs)
            loss = criterion(outputs, ratings.unsqueeze(1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# ============================================================================== #
if __name__ == '__main__':
    trndt, valdt = get_dataloader()
    model = WideAndDeep( wide_dim = len(wide_features), 
                         deep_dim = len(deep_features), 
                         hidden_dim = 32, 
                         output_dim = 1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    from wandb import init ###
    monitor = init( project="recsys", name="recsys_1.0", config = {"version":"v0.1"}) ###

    for epoch in range(100):
        train_loss = train(model, trndt, criterion, optimizer, device)
        test_loss = test(model, valdt, criterion, device)
        print(f'Epoch {epoch+1}/{100} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f}')
        monitor.log({"Train_loss": train_loss, "Test_loss": test_loss}) ###
    monitor.finish() ###

# ============================================================================== #
# *** Movielens training data 80367 as 
#     user_id  item_id    rating  timestamp       age  gender  occupation  \
# 0      195      241 -0.470707   0.351593  1.386383     0.0         0.0   
# 1      195      392  0.417654   0.351642  1.386383     0.0         0.0   
# 2      195      380  0.417654   0.351635  1.386383     0.0         0.0   
# 3      195      250 -0.470707   0.351610  1.386383     0.0         0.0   
# 4      195      654  1.306016   0.351638  1.386383     0.0         0.0   

#    zip_code  release_year  
# 0       415            69  
# 1       415            65  
# 2       415            66  
# 3       415            69  
# 4       415            58  
# *** Movielens validation data 19633 as 
#     user_id  item_id    rating  timestamp       age  gender  occupation  \
# 0      195       12 -1.359069   0.351647  1.386383     0.0         0.0   
# 1      195      761 -0.470707   0.351647  1.386383     0.0         0.0   
# 2      195      172 -1.359069   0.351640  1.386383     0.0         0.0   
# 3      195     1021  0.417654   0.351603  1.386383     0.0         0.0   
# 4      195      844  0.417654   0.351647  1.386383     0.0         0.0   

#    zip_code  release_year  
# 0       415            67  
# 1       415            68  
# 2       415            59  
# 3       415            69  
# 4       415            68  
# wandb: Currently logged in as: xiaogift (xiaolis). Use `wandb login --relogin` to force relogin
# wandb version 0.15.4 is available! To upgrade, please run: , $ pip install wandb --upgrade
# Tracking run with wandb version 0.13.4
# Run data is saved locally in /notebooks/wandb/run-20230703_221704-2fivgpyv
# Syncing run recsys_0.1 to Weights & Biases (docs)
# Epoch 1/100 - Train Loss: 16706.5907 - Test Loss: 13548.8373
# Epoch 2/100 - Train Loss: 11020.0763 - Test Loss: 9055.0114
# Epoch 3/100 - Train Loss: 7326.1761 - Test Loss: 5944.3316
# Epoch 4/100 - Train Loss: 4745.6346 - Test Loss: 3785.6057
# Epoch 5/100 - Train Loss: 2986.7209 - Test Loss: 2351.2358
# Epoch 6/100 - Train Loss: 1848.2076 - Test Loss: 1450.6839
# Epoch 7/100 - Train Loss: 1145.2624 - Test Loss: 902.7475
# Epoch 8/100 - Train Loss: 725.3116 - Test Loss: 581.5146
# Epoch 9/100 - Train Loss: 478.1769 - Test Loss: 392.1693
# Epoch 10/100 - Train Loss: 331.8397 - Test Loss: 278.9962
# Epoch 11/100 - Train Loss: 242.0570 - Test Loss: 207.7651
# Epoch 12/100 - Train Loss: 183.6774 - Test Loss: 159.8029
# Epoch 13/100 - Train Loss: 142.7236 - Test Loss: 124.8630
# Epoch 14/100 - Train Loss: 111.8459 - Test Loss: 97.7510
# Epoch 15/100 - Train Loss: 87.1787 - Test Loss: 75.8778
# Epoch 16/100 - Train Loss: 67.4528 - Test Loss: 58.5375
# Epoch 17/100 - Train Loss: 51.9531 - Test Loss: 44.9910
# Epoch 18/100 - Train Loss: 39.9951 - Test Loss: 34.7311
# Epoch 19/100 - Train Loss: 30.9693 - Test Loss: 27.0051
# Epoch 20/100 - Train Loss: 24.2413 - Test Loss: 21.2462
# Epoch 21/100 - Train Loss: 19.1408 - Test Loss: 16.9356
# Epoch 22/100 - Train Loss: 15.3745 - Test Loss: 13.7157
# Epoch 23/100 - Train Loss: 12.5450 - Test Loss: 11.2841
# Epoch 24/100 - Train Loss: 10.4043 - Test Loss: 9.4353
# Epoch 25/100 - Train Loss: 8.7495 - Test Loss: 7.9987
# Epoch 26/100 - Train Loss: 7.4473 - Test Loss: 6.8698
# Epoch 27/100 - Train Loss: 6.4336 - Test Loss: 5.9704
# Epoch 28/100 - Train Loss: 5.6183 - Test Loss: 5.2624
# Epoch 29/100 - Train Loss: 4.9692 - Test Loss: 4.6832
# Epoch 30/100 - Train Loss: 4.4419 - Test Loss: 4.2066
# Epoch 31/100 - Train Loss: 4.0012 - Test Loss: 3.8159
# Epoch 32/100 - Train Loss: 3.6396 - Test Loss: 3.4889
# Epoch 33/100 - Train Loss: 3.3348 - Test Loss: 3.2168
# Epoch 34/100 - Train Loss: 3.0767 - Test Loss: 2.9761
# Epoch 35/100 - Train Loss: 2.8514 - Test Loss: 2.7720
# Epoch 36/100 - Train Loss: 2.6611 - Test Loss: 2.5950
# Epoch 37/100 - Train Loss: 2.4922 - Test Loss: 2.4363
# Epoch 38/100 - Train Loss: 2.3414 - Test Loss: 2.2982
# Epoch 39/100 - Train Loss: 2.2100 - Test Loss: 2.1760
# Epoch 40/100 - Train Loss: 2.0901 - Test Loss: 2.0588
# Epoch 41/100 - Train Loss: 1.9815 - Test Loss: 1.9573
# Epoch 42/100 - Train Loss: 1.8835 - Test Loss: 1.8643
# Epoch 43/100 - Train Loss: 1.7963 - Test Loss: 1.7792
# Epoch 44/100 - Train Loss: 1.7121 - Test Loss: 1.7031
# Epoch 45/100 - Train Loss: 1.6394 - Test Loss: 1.6348
# Epoch 46/100 - Train Loss: 1.5727 - Test Loss: 1.5649
# Epoch 47/100 - Train Loss: 1.5090 - Test Loss: 1.5055
# Epoch 48/100 - Train Loss: 1.4512 - Test Loss: 1.4547
# Epoch 49/100 - Train Loss: 1.4001 - Test Loss: 1.4024
# Epoch 50/100 - Train Loss: 1.3540 - Test Loss: 1.3556
# Epoch 51/100 - Train Loss: 1.3100 - Test Loss: 1.3141
# Epoch 52/100 - Train Loss: 1.2701 - Test Loss: 1.2761
# Epoch 53/100 - Train Loss: 1.2340 - Test Loss: 1.2386
# Epoch 54/100 - Train Loss: 1.1962 - Test Loss: 1.2039
# Epoch 55/100 - Train Loss: 1.1646 - Test Loss: 1.1723
# Epoch 56/100 - Train Loss: 1.1333 - Test Loss: 1.1424
# Epoch 57/100 - Train Loss: 1.1071 - Test Loss: 1.1168
# Epoch 58/100 - Train Loss: 1.0843 - Test Loss: 1.0939
# Epoch 59/100 - Train Loss: 1.0645 - Test Loss: 1.0759
# Epoch 60/100 - Train Loss: 1.0469 - Test Loss: 1.0577
# Epoch 61/100 - Train Loss: 1.0304 - Test Loss: 1.0423
# Epoch 62/100 - Train Loss: 1.0165 - Test Loss: 1.0288
# Epoch 63/100 - Train Loss: 1.0043 - Test Loss: 1.0168
# Epoch 64/100 - Train Loss: 0.9932 - Test Loss: 1.0055
# Epoch 65/100 - Train Loss: 0.9846 - Test Loss: 0.9962
# Epoch 66/100 - Train Loss: 0.9770 - Test Loss: 0.9889
# Epoch 67/100 - Train Loss: 0.9690 - Test Loss: 0.9811
# Epoch 68/100 - Train Loss: 0.9634 - Test Loss: 0.9773
# Epoch 69/100 - Train Loss: 0.9570 - Test Loss: 0.9755
# Epoch 70/100 - Train Loss: 0.9537 - Test Loss: 0.9659
# Epoch 71/100 - Train Loss: 0.9493 - Test Loss: 0.9613
# Epoch 72/100 - Train Loss: 0.9466 - Test Loss: 0.9590
# Epoch 73/100 - Train Loss: 0.9431 - Test Loss: 0.9547
# Epoch 74/100 - Train Loss: 0.9405 - Test Loss: 0.9530
# Epoch 75/100 - Train Loss: 0.9404 - Test Loss: 0.9547
# Epoch 76/100 - Train Loss: 0.9376 - Test Loss: 0.9516
# Epoch 77/100 - Train Loss: 0.9382 - Test Loss: 0.9484
# Epoch 78/100 - Train Loss: 0.9352 - Test Loss: 0.9457
# Epoch 79/100 - Train Loss: 0.9336 - Test Loss: 0.9525
# Epoch 80/100 - Train Loss: 0.9359 - Test Loss: 0.9455
# Epoch 81/100 - Train Loss: 0.9345 - Test Loss: 0.9469
# Epoch 82/100 - Train Loss: 0.9339 - Test Loss: 0.9548
# Epoch 83/100 - Train Loss: 0.9338 - Test Loss: 0.9503
# Epoch 84/100 - Train Loss: 0.9332 - Test Loss: 0.9442
# Epoch 85/100 - Train Loss: 0.9310 - Test Loss: 0.9535
# Epoch 86/100 - Train Loss: 0.9327 - Test Loss: 0.9424
# Epoch 87/100 - Train Loss: 0.9312 - Test Loss: 0.9433
# Epoch 88/100 - Train Loss: 0.9315 - Test Loss: 0.9427
# Epoch 89/100 - Train Loss: 0.9323 - Test Loss: 0.9417
# Epoch 90/100 - Train Loss: 0.9353 - Test Loss: 0.9424
# Epoch 91/100 - Train Loss: 0.9323 - Test Loss: 0.9436
# Epoch 92/100 - Train Loss: 0.9302 - Test Loss: 0.9416
# Epoch 93/100 - Train Loss: 0.9306 - Test Loss: 0.9449
# Epoch 94/100 - Train Loss: 0.9332 - Test Loss: 0.9412
# Epoch 95/100 - Train Loss: 0.9309 - Test Loss: 0.9396
# Epoch 96/100 - Train Loss: 0.9301 - Test Loss: 0.9446
# Epoch 97/100 - Train Loss: 0.9313 - Test Loss: 0.9402
# Epoch 98/100 - Train Loss: 0.9313 - Test Loss: 0.9395
# Epoch 99/100 - Train Loss: 0.9339 - Test Loss: 0.9437
# Epoch 100/100 - Train Loss: 0.9328 - Test Loss: 0.9412
# Waiting for W&B process to finish... (success).
# VBox(children=(Label(value='0.001 MB of 0.001 MB uploaded (0.000 MB deduped)\r'), FloatProgress(value=1.0, max…
# ,
# Run history:

# Test_loss   █▄▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
# Train_loss  █▄▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁

# Run summary:

# Test_loss   0.94121
# Train_loss  0.93285

# Synced recsys_0.1: https://wandb.ai/xiaolis/recsys/runs/2fivgpyv
# Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
# Find logs at: ./wandb/run-20230703_221704-2fivgpyv/logs