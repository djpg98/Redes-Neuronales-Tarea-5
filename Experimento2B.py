from Dataset import ApproximationDataset
from ActivationFunction import GaussianV2
from cluster import build_rbf_layer_load_centers, build_rbf_layer
from RBFNeuron import RBFNetwork
import math
import os
import sys

#Primer argumento: Nombre del archivo de salida (Debe incluir extensión csv) (Output de la red)
#Segundo argumento: Archivo con centros (Debe incluir extensión npy)
#Tercer argumento: Training error file (Debe incluir extensión csv)

width = math.sqrt(0.01)
phi_function = GaussianV2(width)
center_number = sys.argv[2]

dataset_100 = ApproximationDataset('Spectra100.csv')
#dataset_100 = ApproximationDataset('Spectra100.csv', 100)
dataset_100.to_array()
dataset_real = ApproximationDataset('SpectraReal.csv')
dataset_real.to_array()

#rbf_layer = build_rbf_layer(dataset_100, phi_function, True, 1, 4)
#rbf_layer = build_rbf_layer(dataset_100, phi_function, True, 0.1, 0.5)
rbf_layer = build_rbf_layer_load_centers(sys.argv[2], phi_function)
#rbf_layer = build_rbf_layer(dataset_100, phi_function)

calc = RBFNetwork(rbf_layer)
calc.train_network(dataset_100, 100, 0.001, sys.argv[3])

calc.eval(dataset_100)
calc.eval(dataset_real, sys.argv[1])

os.system('Rscript Plots.R 100 ' + sys.argv[1])