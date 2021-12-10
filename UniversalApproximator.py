import numpy as np

""" Esta clase representa un aproximador universal"""
class UniversalApproximator:

    """ Constructor de la clase
        Parámetros:
            - weights: Peso de cada función en el aproximador
            - functions: Lista de funciones que usa el aproximador. Cada función debe
              tener el mismo índice que su respectivo peso en weights
    """
    def __init__(self, weights, functions):
        self.weights = weights
        self.functions = functions

    """ Produce la salida del aproximador dado un vector de entrada
        Parámetros:
            - input_vector: Vector de entrada
    """
    def output(self, input_vector):

        return (self.weights * np.array([phi.output(input_vector) for phi in self.functions])).sum()