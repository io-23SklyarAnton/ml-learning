import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import MSELoss
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from py_torch_implementation.neural_network.auto_mpg.model import Model

dtype = torch.float32
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(
    "../../../datasets/auto+mpg/auto-mpg.data",
    names=column_names,
    na_values="?",
    comment='\t',
    sep=" ",
    skipinitialspace=True
)
df = df.dropna()
df = df.reset_index(drop=True)

df_train, df_test = sklearn.model_selection.train_test_split(
    df,
    train_size=0.8,
    random_state=1,
)

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']

scaler = StandardScaler()
df_train_norm[numeric_column_names] = scaler.fit_transform(df_train[numeric_column_names])
df_test_norm[numeric_column_names] = scaler.transform(df_test[numeric_column_names])

boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm['Model Year'].values)
df_train_norm['Model Year Bucketed'] = torch.bucketize(
    input=v,
    boundaries=boundaries,
    right=True,
)

v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize(
    input=v,
    boundaries=boundaries,
    right=True,
)

train_one_hot_bucket = one_hot(
    torch.tensor(df_train_norm['Model Year Bucketed'].values)
)
test_one_hot_bucket = one_hot(
    torch.tensor(df_test_norm['Model Year Bucketed'].values)
)

df_train_norm['Origin'] = df_train_norm['Origin'] - 1
df_test_norm['Origin'] = df_test_norm['Origin'] - 1

train_X = torch.cat((
    torch.tensor(
        df_train_norm[['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']].values,
        dtype=dtype
    ),
    torch.tensor(train_one_hot_bucket, dtype=dtype),
    torch.tensor(df_train_norm[['Origin']].values).long()
),
    dim=1
)

test_X = torch.cat((
    torch.tensor(
        df_test_norm[['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']].values,
        dtype=dtype
    ),
    torch.tensor(test_one_hot_bucket, dtype=dtype),
    torch.tensor(df_test_norm[['Origin']].values).long()
),
    dim=1
)

train_y = torch.tensor(df_train[['MPG']].values, dtype=dtype)
test_y = torch.tensor(df_test[['MPG']].values, dtype=dtype)

train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)

train_data_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Model(
    n_features=test_X.shape[1] - 1,
    n_hidden=100,
    embedding_dim=50
)
optimizer = Adam(model.parameters())
loss_f = MSELoss()

for _ in range(30):
    for x_batch, y_batch in train_data_loader:
        optimizer.zero_grad()
        pred = model(x_batch[:, :-1], x_batch[:, -1].long())
        loss = loss_f(pred, y_batch)
        loss.backward()
        optimizer.step()

mae = 0
for x, y in test_data_loader:
    pred = model(x[:, :-1], x[:, -1].long())
    print(f"prediction: {pred.flatten()}, real: {y.flatten()}")
    mae += abs(pred - y)

print(f"avg loss is: {mae / len(test_y)}")

print(df.describe())
