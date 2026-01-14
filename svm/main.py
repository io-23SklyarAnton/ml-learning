from pandas import read_csv
import numpy as np

from scaling.standard_scaling import standard_scale
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

X = data_without_virginica[:, :-1]
y = data_without_virginica[:, 4]

scaled_data = standard_scale(X)

training_X = scaled_data[:-5]
training_y = y[:-5]
unseen_X = scaled_data[-5:]
unseen_y = y[-5:]

_, indices = np.unique(training_y, return_inverse=True)
correct_answers = []

for i in indices:
    y = 1 if i else -1
    correct_answers.append(y)

svm.fit(
    X=training_X,
    y=np.array(correct_answers),
)

for i in range(len(unseen_y)):
    print(
        svm.predict(unseen_X[i]),
        unseen_y[i]
    )
