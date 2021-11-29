from Dataset import ApproximationDataset
from ActivationFunction import GaussianV1
from UniversalApproximator import UniversalApproximator
import numpy as np
import math
import csv
import sys
import os

width = math.sqrt(0.3)

dataset_20 = ApproximationDataset('Spectra20.csv')
dataset_20.to_array()
dataset_real = ApproximationDataset('SpectraReal.csv')
dataset_real.to_array()

function_list = []

for feature, expected_value in dataset_20:
    function_list.append(GaussianV1(feature, width))

function_array = np.array(function_list)
phi_matrix = np.array([[phi.output(feature) for phi in function_array] for feature in dataset_20.features])
inverse = np.linalg.inv(phi_matrix)
weights = np.matmul(inverse, dataset_20.values)

calc = UniversalApproximator(weights, function_array)

with open(sys.argv[1], 'w') as outfile:
    writer = csv.writer(outfile)  
    for feature, expected_value in dataset_real:
        writer.writerow([feature[0], calc.output(feature)])

    outfile.close()

os.system('Rscript Plots.R 20 ' + sys.argv[1])
        

