import numpy as np
from utils import  sample_batch , plot_loss_and_accuracy
import json


# Set a random seed for consistent results
np.random.seed(42)

class Level:
    def __init__(self, inputCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.weights = np.random.rand(outputCount, inputCount) * (2/outputCount)**(1/2) # He initialization
        self.biases = np.zeros((outputCount, 1))

        # Variables pour Adam
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.biases)
        self.v_b = np.zeros_like(self.biases)
        self.t = 0  # Time step

    def feedForward(self, givenInput, activation_func):
        self.input = givenInput
        self.z = np.dot(self.weights, self.input) + self.biases
        self.output = activation_func(self.z)
        return self.output        

    def computeGradient(self, dL_dOutput, activation_func_prime, reg):
        dZ = dL_dOutput * activation_func_prime(self.z)
        dW = np.dot(dZ, self.input.T) + 2 * reg * self.weights  # L2 regularization
        dB = dZ
        dL_dInput = np.dot(self.weights.T, dZ)
        return dW, dB, dL_dInput


    def Gradient_descent(self, dW, dB, eta):
        self.weights -= eta * dW
        self.biases -= eta * dB

    def SGDplusMomentum(self, dW, dB, eta , rho=0.9):
        self.v_w = rho*self.v_w - eta * dW    
        self.v_b = rho*self.v_b - eta * dB

        self.weights += self.v_w
        self.biases += self.v_b

    def adam_optimizer(self, dW, dB, eta, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1  # Update time step

        # Update moments
        self.m_w = beta1 * self.m_w + (1 - beta1) * dW
        self.v_w = beta2 * self.v_w + (1 - beta2) * (dW ** 2)
        self.m_b = beta1 * self.m_b + (1 - beta1) * dB
        self.v_b = beta2 * self.v_b + (1 - beta2) * (dB ** 2)
        
        # Bias correction
        m_w_unbias = self.m_w / (1 - beta1 ** self.t)
        v_w_unbias = self.v_w / (1 - beta2 ** self.t)
        m_b_unbias = self.m_b / (1 - beta1 ** self.t)
        v_b_unbias = self.v_b / (1 - beta2 ** self.t)
        
        # Update weights and biases
        self.weights -= eta * m_w_unbias / (np.sqrt(v_w_unbias) + epsilon)
        self.biases -= eta * m_b_unbias / (np.sqrt(v_b_unbias) + epsilon)    

class NeuralNetwork:
    def __init__(self, NeuronCount , optimizer , loss , file):
        self.NeuronCount = NeuronCount
        self.levels = [Level(NeuronCount[i], NeuronCount[i + 1]) for i in range(len(NeuronCount) - 1)]
        self.optimizer = optimizer
        self.loss = loss
        self.file = file

    def feedForward(self, givenInputs, activation_funcs):
        output = self.levels[0].feedForward(givenInputs, activation_funcs[0])
        for i in range(1, len(self.levels)):
            output = self.levels[i].feedForward(output, activation_funcs[i])
        return output    

    def train(self, X, Y, activation_funcs, activation_funcs_prime, eta, epochs, minibatch ,reg=5e-6 , plot=False):

        n_samples = len(X)
        loss_history = []  # Liste pour stocker les pertes
        accuracy_history = []

        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            X_batch, y_batch = sample_batch(X, Y, n_samples, minibatch)

            for x, y in zip(X_batch, y_batch):
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)

                # Forward pass
                outputs = [x]
                for i in range(len(self.levels)):
                    output = self.levels[i].feedForward(outputs[-1], activation_funcs[i])
                    outputs.append(output)


                # Accumulate the loss
                if self.loss =='mse':
                    total_loss += 0.5 * np.sum((outputs[-1] - y) ** 2)
                elif self.loss == 'cross_entropy_loss':
                    # Cross-entropy loss
                    correct_scores = outputs[-1][np.argmax(y)]  # Correct class scores
                    total_loss += -1 * np.log(correct_scores + 1e-8)[0]  # Add small epsilon to avoid log(0)

                # Backward pass (backpropagation)
                dL_dOutput = outputs[-1] - y
                for i in reversed(range(len(self.levels))):
                    dW, dB, dL_dOutput = self.levels[i].computeGradient(dL_dOutput, activation_funcs_prime[i] , reg )
                    if self.optimizer == "Adam":
                        self.levels[i].adam_optimizer(dW, dB, eta)
                    elif self.optimizer == "SGD":
                        self.levels[i].Gradient_descent(dW, dB, eta)
                    elif self.optimizer == "SGD+Moment":
                        self.levels[i].SGDplusMomentum(dW, dB, eta)  
                           
                                 
                # Calculate correct predictions
                predicted_label = np.argmax(outputs[-1])
                true_label = np.argmax(y)
                if predicted_label == true_label:
                    correct_predictions += 1

            # Calculate accuracy
            accuracy = correct_predictions / minibatch

            # Normalize the loss by batch size
            total_loss /= minibatch

            # Add regularization loss
            reg_loss = reg * sum(np.sum(level.weights ** 2) for level in self.levels)
            total_loss += reg_loss

            # Print loss and accuracy after each epoch
            print(f"Epoch {epoch + 1}/{epochs},\t Loss: {total_loss},\t Accuracy: {accuracy}")

            # Append the loss to the history
            loss_history.append(total_loss)
            accuracy_history.append(accuracy)

            if epoch % 10 == 0 :
                self.save_model(self.file , eta , epochs , total_loss , accuracy )  


        if plot:        
            plot_loss_and_accuracy(epochs , loss_history , accuracy_history)            

        return total_loss , accuracy

    def predict(self, x, activation_func):
        predict_x = np.array([x]).reshape(-1, 1)
        output = self.feedForward(predict_x, activation_func)
        return output
    
    def save_model(self, filename, eta, epochs, final_loss , accuracy ):
        # Créez un dictionnaire pour stocker les poids et les biais de chaque couche ainsi que d'autres informations
        model_data = {
            'NeuronCount': self.NeuronCount,
            'optimizer': self.optimizer,
            'file': self.file,
            'weights': [level.weights.tolist() for level in self.levels],
            'biases': [level.biases.tolist() for level in self.levels],
            'eta': eta,
            'epochs': epochs,
            'loss': final_loss,
            'accuracy':accuracy,
            'loss':self.loss
        }
        
        # Sauvegarde du dictionnaire dans un fichier JSON
        with open(filename, 'w') as f:
            json.dump(model_data, f)
        
        print(f"Modèle sauvegardé sous le nom '{filename}'") 

