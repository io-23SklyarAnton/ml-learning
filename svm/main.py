from pandas import read_csv
import numpy as np

from scaling.standard_scaling import StandardScaler
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

X = data_without_virginica[:, :-1].astype(float)
y = data_without_virginica[:, 4]

training_X = X[:-5]
training_y = y[:-5]
unseen_X = X[-5:]
unseen_y = y[-5:]

standard_scaler = StandardScaler()
scaled_training_X = standard_scaler.fit_transform(training_X)

_, indices = np.unique(training_y, return_inverse=True)
correct_answers = []

for i in indices:
    y = 1 if i else -1
    correct_answers.append(y)

svm.fit(
    X=scaled_training_X,
    y=np.array(correct_answers),
)

scaled_unseen_X = standard_scaler.transform(unseen_X)
for i in range(len(unseen_y)):
    print(
        svm.predict(scaled_unseen_X[i]),
        unseen_y[i]
    )
