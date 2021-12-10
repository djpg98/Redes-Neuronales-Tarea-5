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

""" Implementación de la función logística siguiendo las especificaciones
    de la clase abstracta ActivationFunction. En esta versión, se le debe
    pasar el centro como argumento de inicialización a la clase
"""
class GaussianV1(ActivationFunction):
    """ Contructor de la función
        Parámetros:
            - Center: El centro, que se le restará al argumento que se le
            pasará a la función Gaussiana
            - Width: Valor de la dispersión
    """
    def __init__(self, center, width):
        self.center = center
        self.width = width

    """ Implementación del método output declarado en ActivationFunction
        de acuerdo a la definición de la función gaussiana
        Parámetros:
            - x: Entrada de la función
    """
    def output(self, x):
        arg = np.linalg.norm(x - self.center)
        return math.exp(-1 * math.pow(arg, 2) / (2 * math.pow(self.width, 2)))

    def first_derivative_output(self, x):
        raise Exception("NOT IMPLEMENTED")

    
""" Implementación de la función logística siguiendo las especificaciones
    de la clase abstracta ActivationFunction. En esta versión, no se le debe
    pasar el centro como argumento de inicialización a la clase, ya que
    asume que el módulo del argumento menos el centro es lo que se le
    está pasando a la hora de hacer el cálculo
"""
class GaussianV2(ActivationFunction):

    """ Contructor de la función
        Parámetros:
            - Width: Valor de la dispersión
    """
    def __init__(self, width):
        self.width = width

    """ Implementación del método output declarado en ActivationFunction
        de acuerdo a la definición de la función gaussiana
        Parámetros:
            - x: Entrada de la función. En este caso, si z es el input cuyo valor
            se desea conocer, entonces x=|| z - c || donde c es el centro a utilizar
    """
    def output(self, x):
        return math.exp(-1 * math.pow(x, 2) / (2 * math.pow(self.width, 2)))

    def first_derivative_output(self, x):
        raise Exception("NOT IMPLEMENTED")

""" Implementación de una función lineal siguiendo las especificaciones
    de la clase abstracta ActivationFunction
"""
class Linear(ActivationFunction):

    """ Contructor de la función
        Parámetros:
            - slope: Pendiente de la función lineal
            - intercept: Intercepción de la recta en el eje y
    """
    def __init__(self, slope=1, intercept=0):
        self.slope = slope
        self.intercept = intercept

    """ Implementación del método output declarado en ActivationFunction
        de acuerdo a la definición de una función lineal
        Parámetros:
            - x: Entrada de la función
    """
    def output(self, x):
        return self.slope * x  + self.intercept

    """ Implementación del método first_derivative_output declarado en 
        ActivationFunction de acuerdo a la definición de una función lineal
        Parámetros:
            - x: Entrada de la función
    """
    def first_derivative_output(self, x):
        return self.slope