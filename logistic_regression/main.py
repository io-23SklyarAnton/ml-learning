from sklearn.linear_model import LogisticRegression
from pandas import read_csv
import numpy as np

from implemented_from_scratch.scaling import StandardScaler

data = read_csv("../datasets/iris.csv")

data = np.array(data.values)
np.random.shuffle(data)

X = data[:, :-1].astype(float)
y = data[:, 4:]

training_X = X[:-5]
training_y = y[:-5]
unseen_X = X[-5:]
unseen_y = y[-5:]

standard_scaler = StandardScaler()
scaled_training_X = standard_scaler.fit_transform(training_X)

model = LogisticRegression()
model.fit(
    X=scaled_training_X,
    y=training_y
)

scaled_unseen_X = standard_scaler.transform(unseen_X)
for i in range(len(unseen_y)):
    print(
        model.predict(scaled_unseen_X[i].reshape(1, -1)),
        unseen_y[i]
    )
