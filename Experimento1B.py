from Dataset import ApproximationDataset
from ActivationFunction import GaussianV1
from UniversalApproximator import UniversalApproximator
from metrics import sample_error
import numpy as np
import math
import csv
import sys
import os

#Primer argumento: Nombre del archivo de salida
#Segundo argumento: Parámetro de regularización

width = math.sqrt(0.1)
reg = float(sys.argv[2])

dataset_20 = ApproximationDataset('Spectra20.csv')
dataset_20.to_array()
dataset_real = ApproximationDataset('SpectraReal.csv')
dataset_real.to_array()

function_list = []

for feature, expected_value in dataset_20:
    function_list.append(GaussianV1(feature, width))

function_array = np.array(function_list)
phi_matrix = np.array([[phi.output(feature) for phi in function_array] for feature in dataset_20.features])
inverse = np.linalg.inv(phi_matrix + reg * np.eye(20))
weights = np.matmul(inverse, dataset_20.values)

calc = UniversalApproximator(weights, function_array)

with open(sys.argv[1], 'w') as outfile:
    writer = csv.writer(outfile) 
    total_error = 0 
    for feature, expected_value in dataset_real:
        predicted_output = calc.output(feature)
        writer.writerow([feature[0], predicted_output])
        total_error += sample_error(expected_value, predicted_output)

    mse = total_error / dataset_real.size()
    outfile.close()

print(f"MSE: {mse}")
os.system('Rscript Plots.R 20 ' + sys.argv[1])
        