from pandas import read_csv
import numpy as np

from scaling.standard_scaling import StandardScaler
from k_nearest.find_nearest_neighbor import KNearest

data = read_csv("../iris.csv")

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

k_nearest_model = KNearest(
    k=3,
    training_data=np.hstack((scaled_training_X, training_y))
)

scaled_unseen_X = standard_scaler.transform(unseen_X)
for i in range(len(unseen_y)):
    print(
        k_nearest_model.find(scaled_unseen_X[i]),
        unseen_y[i]
    )
