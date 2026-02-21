import torch
from pandas import read_csv
import numpy as np
from torch import nn, softmax
from torch.utils.data import TensorDataset, DataLoader

from implemented_from_scratch.scaling.standard_scaling import StandardScaler
from py_torch_implementation.neural_network.iris_classificator.model import Model

data = read_csv("../../datasets/iris.csv")

data = np.array(data.values)
np.random.shuffle(data)

X = data[:, :-1].astype(float)
y = data[:, 4]

training_X = X[:-5]
training_y = y[:-5]
unseen_X = X[-5:]
unseen_y = y[-5:]

standard_scaler = StandardScaler()
scaled_training_X = standard_scaler.fit_transform(training_X)

_, indices = np.unique(training_y, return_inverse=True)
correct_answers = []

for i in indices:
    answer = [0, 0, 0]
    answer[i] = 1
    correct_answers.append(answer)

model = Model(
    n_input=4,
    n_hidden=100,
    n_output=3
)

train_dataset = TensorDataset(
    torch.from_numpy(scaled_training_X),
    torch.torch.tensor(correct_answers, dtype=torch.float64),
)
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

unseen_dataset = TensorDataset(
    torch.from_numpy(standard_scaler.transform(unseen_X)),
    unseen_y
)
unseen_dataloader = DataLoader(unseen_dataset, batch_size=1, shuffle=False)

loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(20):
    for x, y in train_data_loader:
        optimizer.zero_grad()

        predict = model(x)
        loss = loss_f(predict, y)
        loss.backward()

        optimizer.step()

for x, y in unseen_dataloader:
    predict = softmax(model(x), dim=1)
    print(f"predict:{predict}, y: {y}")
