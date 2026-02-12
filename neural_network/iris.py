from pandas import read_csv
import numpy as np

from neural_network.activation_functions import ActivationFunction
from neural_network.model import NeuralNetwork, LayerSize
from optimizers import AdamFactory
from scaling.standard_scaling import StandardScaler

data = read_csv("../datasets/iris.csv")

layer_sizes = [
    LayerSize(n_neurons=4),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=3)
]
optimizer_factory = AdamFactory(
    p1=0.9,
    p2=0.999,
    epsilon=1e-8
)
model = NeuralNetwork(
    activation_function=ActivationFunction.RELU,
    layer_sizes=layer_sizes,
    learning_rate=0.001,
    optimizer_factory=optimizer_factory,
)
data = np.array(data.values)
np.random.shuffle(data)

X = data[:, :-1].astype(float)
y = data[:, 4]

training_X = X[:-5]
unseen_X = X[-5:]
training_y = y[:-5]
unseen_y = y[-5:]

_, indices = np.unique(training_y, return_inverse=True)
correct_answers = []

for i in indices:
    answer = [0, 0, 0]
    answer[i] = 1
    correct_answers.append(answer)

standard_scaler = StandardScaler()
scaled_training_X = standard_scaler.fit_transform(training_X)

model.fit(
    X=scaled_training_X,
    Y=np.array(correct_answers),
    n_epochs=100
)

scaled_unseen_X = standard_scaler.transform(unseen_X)
for x, y in zip(scaled_unseen_X, unseen_y):
    print(
        model.predict(x),
        y
    )
