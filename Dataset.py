import csv
import random
import numpy as np

""" Conjunto de funciones básicas que debe tener un dataset """
class DatasetMixin:

    """ Añade una componente de sesgo a cada dato en el dataset. Esto se logra
        agregando una componente adicional que siempre vale 1 a todos los datos 
        en el dataset.
    """
    def add_bias_term(self):

        for i in range(len(self.features)):

            self.features[i] = np.concatenate((np.ones(1), self.features[i]))

        self.features = np.array(self.features)
        self.values = np.array(self.values)

    """ Devuelve la cantidad de elementos en el dataset """
    def size(self):

        return len(self.features)

    """ Devuelve la cantidad de elementos pertenecientes
        al conjunto de entrenamiento del dataset """
    def training_data_size(self):

        return len(self.training_data)

    """ Devuelve la cantidad de elementos pertenecientes
        al conjunto de validación del dataset """
    def validation_data_size(self):

        return len(self.validation_data)

    """ Devuelve el tamaño del input vector (Incluendo el término de bias si
        este ha sido agregado
    """
    def feature_vector_length(self):

        return len(self.features[0])

    """ Iterador para todos los elementos del dataset """
    def __iter__(self):

        for pair in zip(self.features, self.values):

            yield pair

    """ Iterador para el conjunto de datos de entrenamiento
        del dataset
    """
    def training_data_iter(self):

        for index in self.training_data:

            yield (self.features[index], self.values[index])

    """ Iterador para el conjunto de datos de validación
        del dataset
    """
    def validation_data_iter(self):

        for index in self.validation_data:

            yield (self.features[index], self.values[index])

    """ Altera aleatoriamente el orden en que se iteran los elementos del
        conjunto de datos de entrenamiento
    """
    def shuffle_training_data(self):
        random.shuffle(self.training_data)

""" Clase que hereda de DatasetMixin utilizada para resolver el problema de
    aproximación de la tarea 5
"""
class ApproximationDataset(DatasetMixin):

    """ Constructor de la clase:
        Parámetros:
            - datafile: Archivo csv de donde se extrae la información
            - cente_number: Número de datos a utilizar como centros

    """
    def __init__(self, datafile, center_number=None):

        self.features = []
        self.values = []

        with open(datafile, 'r') as csv_file:

            data_reader = csv.reader(csv_file, delimiter=",")

            for row in data_reader:

                features, value = row[:-1], row[-1:][0]

                self.features.append(np.array(list(map(float, features))))
                self.values.append(float(value))

            csv_file.close()

        index_list = [i for i in range(len(self.features))]
        self.training_data = random.sample(index_list, int(0.80 * len(self.features)))
        self.validation_data = [index for index in index_list if index not in self.training_data]

        if center_number is not None:

            self.center_points = random.sample(index_list, center_number)
            self.cluster_points = [index for index in index_list if index not in self.center_points]
        else:
            self.center_points = []
            self.cluster_points = []

    """ Devuelve un vector con los datos a utilizar como centros """
    def center_set(self):

        return np.array([self.features[index] for index in self.center_points])

    """ Devuelve un punto aleatorio que no sea centro """
    def random_cluster_point(self):

        index = random.randint(0, len(self.cluster_points))
        return self.features[index]
            
    """ Vuelve las listas de features y values arrays de numpy para optimizar """
    def to_array(self):

        self.features = np.array(self.features)
        self.values = np.array(self.values)