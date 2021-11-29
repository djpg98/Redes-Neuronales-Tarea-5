from abc import ABC, abstractmethod
import numpy as np
import math

""" Esta clase abstracta declara los métodos que debe implementar cualquier
    función que vaya a ser utilizada como función de activación
"""
class ActivationFunction(ABC):

    """ Este método devuelve el output de la función de activación dado
        un input x
        parámetros:
            - x: Entrada de la función
    """
    @abstractmethod
    def output(self, x):
        pass

    """ Este método devuelve el output de la primera derivada de la función 
        de activación dado un input x
        parámetros:
            - x: Entrada de la función
    """
    @abstractmethod
    def first_derivative_output(self, x):
        pass

""" Implementación de la función logística siguiendo las especificaciones
    de la clase abstracta ActivationFunction
"""
class Logistic(ActivationFunction):

    """ Contructor de la función
        Parámetros:
            - alpha (Opcional): Factor de escalamiento por el cual se multiplica
            el input. Si no se le pasa, el default es 1
    """
    def __init__(self, alpha=1):
        self.alpha=alpha

    """ Implementación del método output declarado en ActivationFunction
        de acuerdo a la definición de la función logística
        Parámetros:
            - x: Entrada de la función
    """
    def output(self, x):
        return 1 / (1 + math.exp(-self.alpha * x))

    """ Implementación del método first_derivative_output declarado en 
        ActivationFunction de acuerdo a la definición de la función logística
        Parámetros:
            - x: Entrada de la función
    """
    def first_derivative_output(self, x):
        output_val = self.output(x)
        return self.alpha * output_val * (1 - output_val)

class GaussianV1(ActivationFunction):
    def __init__(self, center, width):
        self.center = center
        self.width = width

    def output(self, x):
        arg = np.linalg.norm(x - self.center)
        return math.exp(-1 * math.pow(arg, 2) / (2 * math.pow(self.width, 2)))

    def first_derivative_output(self, x):
        raise Exception("NOT IMPLEMENTED")

    

class GaussianV2(ActivationFunction):

    def __init__(self, width):
        self.width = width

    def output(self, x):
        return math.exp(-1 * math.pow(x, 2) / (2 * math.pow(self.width, 2)))

    def first_derivative_output(self, x):
        raise Exception("NOT IMPLEMENTED")