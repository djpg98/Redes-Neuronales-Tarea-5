from RBFNeuron import RBFNeuron
import numpy as np
import copy

""" Ajusta uno de los centros utilizados por una RBF para que se acerque más al 
    datapoint que no es centro más cercano
    Parámetros:
        - dataset: Dataset del que se extraen los datos
        - center_set: Lista actual de centros
        - alpha: Factor de escalamiento

"""
def center_adjust(dataset, center_set, alpha):
    data_point = dataset.random_cluster_point()
    results = np.array([np.linalg.norm(data_point - center) for center in center_set])
    index = np.argmin(results)
    center_set[index] += (alpha  * (data_point - center_set[index]))

""" Algoritmo de conglomerado que realiza ajustes iterativos al conjunto de centros 
    de una RBF hasta que se alcanza una determinada tolerancia. Acerca los centros hasta
    la media del cluster más cercano
    Parámetros:
        - dataset: Dataset del que se extraen los datos
        - center_set: Lista inicial de centros
        - tolerance: Tolerancia hasta la cual el algoritmo iterará
        - alpha: Factor de escalamiento

"""
def cluster_select(dataset, center_set, tolerance, alpha):

    old_center = copy.deepcopy(center_set)
    center_adjust(dataset, center_set, alpha)

    while (np.linalg.norm(center_set - old_center) < tolerance):
        old_center = copy.deepcopy(center_set)
        center_adjust(dataset, center_set, alpha)

""" Crea una lista aleatoria de centros y dependiendo los parámetros pudede
    aplicar un algoritmo de conglomerados para ajustar los mismos
    Parámetros:
        - dataset: Dataset del que se extraen los datos
        - use_cluster: Indica si ajustar los centros mediante el algoritmo de
          conglomerados
        - tolerance: Tolerancia hasta la cual el algoritmo iterará
        - alpha: Factor de escalamiento

"""
def center_select(dataset, use_cluster=False, tolerance=None, alpha=None):

    center_set = dataset.center_set()
    if use_cluster:
        cluster_select(dataset, center_set, tolerance, alpha)

    with open('centers.txt', 'a') as f:
        f.write(str(len(center_set)) + '\n')
        f.write(str(center_set))
        f.write('\n\n')

    return center_set

""" Construye la capa escondida de una RBF
    Parámetros:
        - dataset: Dataset del que se extraen los datos
        - phi_function: Función phi a utilizar por la RBF. Debe ser una implementación
          de la clase abstracta ActivationFunction
        - use_cluster: Indica si ajustar los centros mediante el algoritmo de
          conglomerados
        - tolerance: Tolerancia hasta la cual el algoritmo iterará
        - alpha: Factor de escalamiento
"""
def build_rbf_layer(dataset, phi_function, use_cluster=False, tolerance=None, alpha=None):

    center_set = center_select(dataset, use_cluster, tolerance, alpha)

    rbf_layer = []
    for center in center_set:
        rbf_layer.append(RBFNeuron(center, phi_function))

    return rbf_layer

"""
    Construye la capa escondida de una RBF a partir de una lista de centros
    Parámetros:
        - center_file: Archivo npy que contiene el vector con los centros a usar
        - phi_function: Función phi a utilizar por la RBF. Debe ser una implementación
          de la clase abstracta ActivationFunction        
"""
def build_rbf_layer_load_centers(center_file, phi_function):

    center_set = np.load(center_file)

    rbf_layer = []
    for center in center_set:
        rbf_layer.append(RBFNeuron(center, phi_function))

    return rbf_layer



