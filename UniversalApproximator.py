import numpy as np

class UniversalApproximator:

    def __init__(self, weights, functions):
        self.weights = weights
        self.functions = functions

    def output(self, input_vector):

        return (self.weights * np.array([phi.output(input_vector) for phi in self.functions])).sum()