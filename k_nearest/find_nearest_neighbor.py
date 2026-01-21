import numpy as np


class KNearest:
    def __init__(
            self,
            k: int,
            training_data: np.ndarray,
    ):
        self._k = k
        self._training_data = training_data
        self._n = len(self._training_data[0])

    def find(
            self,
            x: np.ndarray,
    ):
        euclidean_distances: list = []

        for sample in self._training_data:
            distance: float = 0
            for i in range(self._n - 1):
                distance += (sample[i] - x[i]) ** 2

            distance = np.sqrt(distance)

            euclidean_distances.append([distance, sample[-1]])

        np_euclidean_distances = np.array(euclidean_distances)
        sorted_indices = np.argsort(np_euclidean_distances[:, 0])
        np_euclidean_distances_sorted = np_euclidean_distances[sorted_indices]

        k_nearest = np_euclidean_distances_sorted[:self._k + 1]

        unique_strings, counts = np.unique(k_nearest[:, 1], return_counts=True)
        max_index = np.argmax(counts)
        most_frequent_sample = unique_strings[max_index]

        return most_frequent_sample