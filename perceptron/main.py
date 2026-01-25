from pandas import read_csv
import numpy as np

from perceptron.perceptron import Perceptron

data = read_csv("../iris.csv")

perceptron = Perceptron(layer_sizes=[4, 3])
data = np.array(data.values)
np.random.shuffle(data)

training_data = data[:-5]
unseen_data = data[-5:]

_, indices = np.unique(training_data[:, 4], return_inverse=True)
correct_answers = []

for i in indices:
    answer = [-1, -1, -1]
    answer[i] = 1
    correct_answers.append(answer)

perceptron.fit_model(
    epochs=100,
    features=training_data[:, :-1],
    correct_answers=np.array(correct_answers),
)

for case in unseen_data:
    print(
        perceptron.predict(case[:-1]),
        case[-1]
    )
