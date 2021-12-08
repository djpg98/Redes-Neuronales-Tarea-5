from metrics import sample_error
from Adaline import Adaline
import numpy as np
import csv

class RBFNeuron:

    def __init__(self, center, phi_function):

        self.center = center
        self.phi_function = phi_function

    def output(self, input_vector):

        if (len(input_vector) == 1):
            return self.phi_function.output(abs(input_vector[0] - self.center[0]))
        else:
            return self.phi_function.output(np.linalg.norm(input_vector - self.center))

class RBFNetwork:

    def __init__(self, rbf_list):

        self.rbf_list = np.array(rbf_list)
        self.adaline = Adaline(len(rbf_list))

    def hidden_layer_output(self, input_vector):
        return np.concatenate((np.ones(1), np.array([neuron.output(input_vector) for neuron in self.rbf_list])))

    def output(self, input_vector):
        adaline_input = np.concatenate((np.ones(1), np.array([neuron.output(input_vector) for neuron in self.rbf_list])))
        return self.adaline.output(adaline_input)

    def train_network(self, dataset, epochs, learning_rate, save_training_error=''):

        if save_training_error != '':
            training_data = [["epoch", "error"]]
        
        print("Training information\n")
        print("Epoch, error")

        for current_epoch in range(epochs): #Para cada epoch

            sum_mse = 0 #Aquí se va acumulando el error para cada muestra

            for features, expected_value in dataset.training_data_iter(): #Se itera sobre las muestras en el dataset

                adaline_input = self.hidden_layer_output(features)

                output_value = self.adaline.output(adaline_input) #Se produce el output dados los features (Utilizando la función lineal)

                error = sample_error(expected_value, output_value)

                self.adaline.adjust_weights(expected_value, output_value, learning_rate, adaline_input) #Se asjustan los pesos de acuerdo al gradiente

                sum_mse += error #Actualizar error total

            mse = sum_mse / dataset.training_data_size() #Dividie el error entre el número de muestras para tener el error promedio

            dataset.shuffle_training_data() #Cambiar el orden en que se muestran los datos
            print(f'{current_epoch + 1}, {mse}')

            if save_training_error != '':
                training_data.append([current_epoch + 1, mse])


        if save_training_error != '': #Escribir en un archivo el error cometido en cada epoch

            with open(save_training_error, 'w') as training_results:
                writer = csv.writer(training_results)

                for row in training_data:
                    writer.writerow(row)

                training_results.close()

    """ Devuelve la precision y la accuracy para un dataset test
        Parámetros:
            - Dataset: Instancia de una clase que hereda el mixin DatasetMixin que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el mismo
    """
    def eval(self, dataset, save_output=''):

        print("Test information\n")
        print("error")

        if save_output != '':
            output_header = [['in_val', 'out_val']]
            output_list = []

        sum_mse = 0

        if save_output == '':
            for features, expected_value in dataset.validation_data_iter():

                output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)

                error = sample_error(expected_value, output_value)
                sum_mse += error

                if save_output != '':
                    output_list.append([features[0], output_value])

            mse = sum_mse / dataset.validation_data_size()
        else:
            for features, expected_value in dataset:

                output_value = self.output(features) #Se produce el output dados los features (Utilizando la función lineal)

                error = sample_error(expected_value, output_value)
                sum_mse += error

                if save_output != '':
                    output_list.append([features[0], output_value])

            mse = sum_mse / dataset.size()
            
            
        print(f'{mse}')

        if save_output != '': #Escribir en un archivo el error cometido en cada epoch

            output_list.sort(key=lambda x: x[0])
            complete_output_list = output_list

            with open(save_output, 'w') as eval_results:
                writer = csv.writer(eval_results)

                for row in complete_output_list:
                    writer.writerow(row)

                eval_results.close()

