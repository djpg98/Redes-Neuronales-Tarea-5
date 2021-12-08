from RBFNeuron import RBFNeuron
import numpy as np
import copy

def center_adjust(dataset, center_set, alpha):
    data_point = dataset.random_cluster_point()
    results = np.array([np.linalg.norm(data_point - center) for center in center_set])
    index = np.argmin(results)
    center_set[index] += (alpha  * (data_point - center_set[index]))

def cluster_select(dataset, center_set, tolerance, alpha):

    old_center = copy.deepcopy(center_set)
    center_adjust(dataset, center_set, alpha)

    while (np.linalg.norm(center_set - old_center) < tolerance):
        old_center = copy.deepcopy(center_set)
        center_adjust(dataset, center_set, alpha)

def center_select(dataset, use_cluster=False, tolerance=None, alpha=None):

    center_set = dataset.center_set()
    if use_cluster:
        cluster_select(dataset, center_set, tolerance, alpha)

    with open('centers.txt', 'a') as f:
        f.write(str(len(center_set)) + '\n')
        f.write(str(center_set))
        f.write('\n\n')

    return center_set

def build_rbf_layer(dataset, phi_function, use_cluster=False, tolerance=None, alpha=None):

    center_set = center_select(dataset, use_cluster, tolerance, alpha)

    rbf_layer = []
    for center in center_set:
        rbf_layer.append(RBFNeuron(center, phi_function))

    return rbf_layer

def build_rbf_layer_load_centers(center_file, phi_function):

    center_set = np.load(center_file)

    rbf_layer = []
    for center in center_set:
        rbf_layer.append(RBFNeuron(center, phi_function))

    return rbf_layer



