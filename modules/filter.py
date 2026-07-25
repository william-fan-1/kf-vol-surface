import numpy as np

class KalmanFilter(object):
    def __init__(self, F = None, Z = None, Q = None, eps = None, P = None, x0 = None):

        if(F is None or Z is None):
            raise ValueError('Set proper system dynamics.')

        self.n = F.shape[1]
        self.m = Z.shape[1]

        # The transition matrix
        self.F = F

        # Observation model
        self.Z = Z

        # Process noise - Hidden Pattern
        self.Q = np.eye(self.n) if Q is None else Q

        # Measurement noise
        self.eps = np.eye(self.n) if eps is None else eps

        # Error covariance
        self.P = np.eye(self.n) if P is None else P

        # state estimate
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) 
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.Z, self.x)
        S = self.eps + np.dot(self.Z, np.dot(self.P, self.Z.T))
        K = np.dot(np.dot(self.P, self.Z.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.Z), self.P), 
        	(I - np.dot(K, self.Z)).T) + np.dot(np.dot(K, self.eps), K.T)