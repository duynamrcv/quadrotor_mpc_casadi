import numpy as np

# RBF Neural Network Class
class RBFNet:
    def __init__(self, num_inputs1, num_inputs2, num_rbfs, output_dim, learning_rate=0.01):
        # Initialize centers, widths, and weights
        self.centers1 = np.array([np.linspace(-10, 10, num_rbfs) for _ in range(num_inputs1)]).T   # Centers of RBFs
        self.centers2 = np.array([np.linspace(-10, 10, num_rbfs) for _ in range(num_inputs2)]).T   # Centers of RBFs
        self.sigmas = 5.0  # Widths of RBFs
        self.eta = 0.5
        self.beta = 0.1
        self.weights = np.random.randn(num_rbfs, output_dim)  # Weights from RBF layer to output
        self.learning_rate = learning_rate

        self.weights1 = self.weights
        self.weights2 = self.weights

    # Forward pass
    def predict(self, x1, x2, error):
        rbf_activations = np.array([self.rbf(x1, x2, self.centers1[i], self.centers2[i], self.sigmas)
                                     for i in range(len(self.centers1))])

        # Update weight
        weights_dot =  self.learning_rate * (np.outer(rbf_activations, error) - 
                                             self.eta * np.linalg.norm(error) * self.weights)
        
        self.weights += weights_dot + self.beta*(self.weights1 - self.weights2)
        self.weights2 = self.weights1
        self.weights1 = self.weights
        
        output = np.dot(rbf_activations, self.weights)  # Linear combination of RBF outputs
        return output
    
    @staticmethod
    # Radial Basis Function (Gaussian)
    def rbf(x1, x2, c1, c2, s):
        return np.exp(-(np.linalg.norm(x1 - c1) ** 2 + np.linalg.norm(x2 - c2) ** 2) / (s ** 2))
