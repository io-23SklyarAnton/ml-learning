from pandas import read_csv
import numpy as np

from neural_network.activation_functions import ActivationFunction
from neural_network.model import NeuralNetwork, LayerSize
from optimizers import AdamFactory
from scaling.minmax_scaling import MinMaxScaler

data = read_csv('../mnist_784.csv')

layer_sizes = [
    LayerSize(n_neurons=784),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=50),
    LayerSize(n_neurons=10)
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
y = data[:, 784]

unseen_count = 100
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
    n_epochs=1
)

scaled_unseen_X = standard_scaler.transform(unseen_X)

correct_count = 0
for x, y in zip(scaled_unseen_X, unseen_y):
    prediction = model.predict(x)
    answer = np.argmax(prediction)
    correct_count += int(answer == y)

print(f"success percent is {correct_count / unseen_count * 100}")
