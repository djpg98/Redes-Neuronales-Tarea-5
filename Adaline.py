from Perceptron import Perceptron
from ActivationFunction import Linear
from metrics import accuracy, precision, sample_error
import numpy as np
import csv

""" Subclase de la clase Perceptron con las adaptaciones necesarias para que un
    Perceptrón genérico pase a ser un Adaline. Nótese que la diferencia fundamental
    es que el Adaline utiliza la función lineal para entrenar, y si es necesario
    hacer una clasificación binaria, utiliza una función (Normalmente la umbral)
    a la hora de evaluar
"""
class Adaline(Perceptron):

    """ Método constructor de la clase. Básicamente llama al constructor de Perceptron
        y agrega dos campos: weights_gradient, que almacena la dirección contraria al
        gradiente (Y permitiría verificar si se alcanzó un mínimo) y threshold_function,
        que permite agregar una función threshold_adicional a la de activación a la 
        hora de hacer la evaluación (No el entrenamiento), si se requiriera hacer una
        clasificación binaria
        Parámetros:
            - input_dimension: Cantidad de inputs que recibe el perceptron (Sin contar el sesgo)
            - threshold_function: Función que permite hacer una clasificación binaria para un
                input durante la etapa de prueba o evaluación
    """
    def __init__(self, input_dimension, threshold_function=None):
        super().__init__(input_dimension, Linear())
        self.weights_gradient = np.array([0 for i in range(input_dimension + 1)])
        self.threshold_function=threshold_function

    """ Permite ajustar los pesos del perceptron cuando hay un dato mal clasificado 
        Parámetros:
            - expected_value: Valor que se esperaba devolviera el perceptron para el dato dado
            - output_value: Valor devuelto por el perceptron para el dato dado
            - learning_rate: Tasa de aprendizaje a aplicar
            - features: El dato a partir del cual se obtuvo el resultado de output_value
    """
    def adjust_weights(self, expected_value, output_value, learning_rate, features):

        factor = learning_rate * (expected_value - output_value)

        self.weights_gradient = factor * features

        self.weights = self.weights + self.weights_gradient

    """ Utiliza el output de la función de activación (La función lineal) como input de la función
        self.threshold, para poder realizar una clasificación de los datos. Esto se utiliza, por ejemplo
        al momento de hacer la evaluación del conjunto test en la pregunta 2, ya que la idea ahí era tener
        un vector de -1 y 1, por lo que hacía falta aplicar una función umbral.
        Parámtros:
            - inputs: Vector que actúa como input del Adaline 
    """
    def output_with_threshold(self, inputs):
        return self.threshold_function(self.activation_function(self.sum_inputs(inputs)))

    """ Implementación del algoritmo LMS para un único Adaline:
        Parámetros:
            - dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen tres: BinaryDataset, MultiClassDataset y PolinomicalDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_training_error: Nombre de un archivo csv en el que se guardará el error promedio
              para cada epoch
    """
    #Fix this one when you know how to train for RBF
    def train(self, dataset, epochs, learning_rate, verbose=False, save_training_error=''):
        #No hace falta hacer dataset.add_bias() porque en en la pregunta 3
        #el bias se añade al momento de crear el vector de características 
        #Apartir del dato, pero si se fuera a usar otro Dataset que no
        #sea de clase PolinomicalDataset, debe agregarse aquí
        assert(dataset.feature_vector_length() == len(self.weights))

        if save_training_error != '':
            training_data = [["epoch", "error"]]

        print("Training information\n")
        print("Epoch, error")

        for current_epoch in range(epochs): #Para cada epoch

            sum_mse = 0 #Aquí se va acumulando el error para cada muestra

            for features, expected_value in dataset: #Se itera sobre las muestras en el dataset

                output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)
                
                error = sample_error([expected_value], [output_value]) #Se calcula el error para este sample

                self.adjust_weights(expected_value, output_value, learning_rate, features) #Se asjustan los pesos de acuerdo al gradiente

                sum_mse += error #Actualizar error total

            mse = sum_mse / dataset.size() #Dividie el error entre el número de muestras para tener el error promedio

            dataset.shuffle_all() #Cambiar el orden en que se muestran los datos
            if verbose:
                print(f'{current_epoch + 1}, {mse}')

            if save_training_error != '':
                training_data.append([current_epoch + 1, mse])


        if verbose:
            print("Pesos finales: ")
            print(self.weights)
            print("")

        if save_training_error != '': #Escribir en un archivo el error cometido en cada epoch

            with open(save_training_error, 'w') as training_results:
                writer = csv.writer(training_results)

                for row in training_data:
                    writer.writerow(row)

                training_results.close()


    """ Devuelve la precision y la accuracy para un dataset test
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin (En esta tarea
              existen tres: BinaryDataset, MultiClassDataset y PolinomicalDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
    """
    def eval(self, dataset, save_output=''):

        print("Test information\n")
        print("error")

        if save_output != '':
            output_header = [['in_val', 'out_val']]
            output_list = []

        sum_mse = 0

        for features, expected_value in dataset:

            output_value = self.output(features)
            error = sample_error([expected_value], [output_value])
            sum_mse += error

            if save_output != '':
                output_list.append([features[1],output_value])

        mse = sum_mse / dataset.size()
            
        print(f'{mse}')

        if save_output != '': #Escribir en un archivo el error cometido en cada epoch

            output_list.sort(key=lambda x: x[0])
            complete_output_list = output_header + output_list

            with open(save_output, 'w') as eval_results:
                writer = csv.writer(eval_results)

                for row in complete_output_list:
                    writer.writerow(row)

                eval_results.close()