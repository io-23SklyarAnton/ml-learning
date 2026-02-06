from pandas import read_csv
import numpy as np

from cnn.model import CNN
from cnn import layers
from optimizers import AdamFactory
from scaling.minmax_scaling import MinMaxScaler

data = read_csv('../mnist_784.csv')

optimizer_factory = AdamFactory(
    p1=0.9,
    p2=0.999,
    epsilon=1e-8
)
learning_rate = 0.001
layers_architecture = [
    layers.Convolution(
        optimizer_factory=optimizer_factory,
        learning_rate=learning_rate,
        filter_size=3,
        in_channels=1,
        out_channels=10
    ),
    layers.ReLU(),
    layers.MaxPooling(pool_size=2),

    layers.Convolution(
        optimizer_factory=optimizer_factory,
        learning_rate=learning_rate,
        filter_size=3,
        in_channels=10,
        out_channels=20
    ),
    layers.ReLU(),
    layers.MaxPooling(pool_size=2),

    layers.Flatten(),

    layers.Dense(
        optimizer_factory=optimizer_factory,
        learning_rate=learning_rate,
        in_features=20 * 5 * 5,
        out_features=64,
    ),
    layers.ReLU(),

    layers.Dense(
        optimizer_factory=optimizer_factory,
        learning_rate=learning_rate,
        in_features=64,
        out_features=10,
    ),
    layers.Softmax()
]
model = CNN(layers_architecture)

data = np.array(data.values)
np.random.shuffle(data)

X = data[:, :-1].astype(float)
y = data[:, 784]

X = X.reshape(-1, 1, 28, 28)

unseen_count = 1000
training_X = X[:-unseen_count]
unseen_X = X[-unseen_count:]
training_y = y[:-unseen_count]
unseen_y = y[-unseen_count:]

correct_answers = []

for i in training_y:
    answer = [0] * 10
    answer[i] = 1
    correct_answers.append(answer)

standard_scaler = MinMaxScaler()
scaled_training_X = standard_scaler.fit_transform(training_X)

model.fit(
    X=scaled_training_X,
    Y=np.array(correct_answers),
    n_epochs=2
)

scaled_unseen_X = standard_scaler.transform(unseen_X)

correct_count = 0
for x, y in zip(scaled_unseen_X, unseen_y):
    prediction = model.predict(x)
    answer = np.argmax(prediction)
    correct_count += int(answer == y)

print(f"success percent is {correct_count / unseen_count * 100}")
