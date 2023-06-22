import pandas as pd
from glob import glob
from tqdm import tqdm

from utils import settings
from calcuate import calc_velocity

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

pos_paths = glob("../data/raw/position/*")

df = pd.DataFrame()
for pos_path in pos_paths:
    tmp = pd.read_csv(pos_path)
    tmp = calc_velocity(tmp)
    df = pd.concat([df, tmp], axis=0)

input_columns = [
    "Head.x", "Head.y", "Head.z",
    "Hand_L.x", "Hand_L.y", "Hand_L.z",
    "Hand_R.x", "Hand_R.y", "Hand_R.z",
    "Head.vx", "Head.vy", "Head.vz",
    "Hand_L.vx", "Hand_L.vy", "Hand_L.vz",
    "Hand_R.vx", "Hand_R.vy", "Hand_R.vz",
]

output_columns = settings.POSITIONCOLUMNS

# model定義
input_size = len(input_columns)
hidden_size = 100
output_size = len(output_columns)

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out


model = MLP(input_size, hidden_size, output_size)

# 損失関数と最適化アルゴリズムの定義
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = df[input_columns]
y = df[output_columns]

# Tensor化
x = torch.tensor(x.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

# DataSetの作成
dataset = torch.utils.data.TensorDataset(x, y)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaderの作成
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

epochs = 1
with tqdm(range(epochs)) as t:
    for epoch in t:
        for inputs, labels in train_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 勾配の初期化と逆伝播
            optimizer.zero_grad()
            loss.backward()
            # パラメータの更新
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            # 右に表示
            t.set_postfix(loss=loss.item())

# 推論
model.eval()  # 推論モードに切り替える

with torch.no_grad():
    predictions = []
    error = 0
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        predictions.append(outputs)
        loss = criterion(outputs, labels)
        error += loss.item()
    print(error)

df_test = pd.read_csv("../data/raw/position/dataset-1_walk_giant_001_pos.csv")
df_test = calc_velocity(df_test)
X = torch.Tensor(df_test[input_columns].values)
y: pd.DataFrame = df_test[settings.POSITIONCOLUMNS]
outputs: torch.Tensor = model(X)

from animation import create_gif

# create_gif(outputs, "walk_giant_001")
create_gif("dataset-1_walk_giant_001_pos.csv")
# create_gif(y, outputs, "walk_giant_001")
