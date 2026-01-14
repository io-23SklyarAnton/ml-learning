from pandas import read_csv
import numpy as np

from svm.model import SVMModel

data = read_csv("../iris.csv")

svm = SVMModel(
    C=1.0,
    max_passes=10,
    gamma=0.1,
    tol=0.01,
)
data = np.array(data.values)
virginica_mask = np.all(data != 'virginica', axis=1)
data_without_virginica = data[virginica_mask]

np.random.shuffle(data_without_virginica)

training_data = data_without_virginica[:-5]
unseen_data = data_without_virginica[-5:]

_, indices = np.unique(training_data[:, 4], return_inverse=True)
correct_answers = []

for i in indices:
    y = 1 if i else -1
    correct_answers.append(y)

svm.fit(
    X=training_data[:, :-1],
    y=np.array(correct_answers),
)

for case in unseen_data:
    print(
        svm.predict(case[:-1]),
        case[-1]
    )
