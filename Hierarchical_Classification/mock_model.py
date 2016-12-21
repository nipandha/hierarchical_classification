import numpy as np

class MockModel:
    def __init__(self, y):
        self.y = y

    def predict(self, X):
        length = X.shape[0]

        return np.array([self.y] * length)