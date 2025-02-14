import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt



def save_model(nn, filename, eta, epochs, final_loss , accuracy):
    # Créez un dictionnaire pour stocker les poids et les biais de chaque couche ainsi que d'autres informations
    model_data = {
        'NeuronCount': nn.NeuronCount,
        'optimizer': nn.optimizer,
        'weights': [level.weights.tolist() for level in nn.levels],
        'biases': [level.biases.tolist() for level in nn.levels],
        'eta': eta,
        'epochs': epochs,
        'loss': final_loss,
        'accuracy':accuracy
    }
    
    # Sauvegarde du dictionnaire dans un fichier JSON
    with open(filename, 'w') as f:
        json.dump(model_data, f)
    
    print(f"Modèle sauvegardé sous le nom '{filename}'")



def load_model(NeuralNetwork, filename):
    with open(filename, 'r') as f:
        model_data = json.load(f)
    
    # Reconstruct the neural network using the loaded data
    nn = NeuralNetwork(model_data['NeuronCount'], model_data['optimizer'] , model_data['loss'] , model_data['file'])
    
    for level, weights, biases in zip(nn.levels, model_data['weights'], model_data['biases']):
        level.weights = np.array(weights)
        level.biases = np.array(biases)
    
    print(f"Modèle chargé depuis '{filename}'")
    print(f"Learning rate: {model_data['eta']}, Epochs: {model_data['epochs']}, Loss: {model_data['loss']} , accuracy: {model_data['accuracy']}")

    return nn


def load_and_preprocess_image(image_path, target_size):
    # Charger l'image
    img = Image.open(image_path)
    
    # Convertir en RGB si l'image n'a pas trois canaux
    img = img.convert('RGB')
    
    # Redimensionner l'image
    img = img.resize(target_size)
    
    # Convertir en tableau numpy
    img_array = np.array(img)
    
    # Normaliser les pixels (0-255 -> 0-1)
    img_array = img_array.astype('float32') / 255.0
    
    # Ajouter une dimension pour le lot (batch size)
    img_array = np.expand_dims(img_array, axis=0)  # Shape devient (1, target_size[0], target_size[1], 3)
    
    return img_array

def kaiming_initializer(Din, Dout, K=None, relu=True):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions for
      this layer
    - K: If K is None, then initialize weights for a linear layer with Din input
      dimensions and Dout output dimensions. Otherwise if K is a nonnegative
      integer then initialize the weights for a convolution layer with Din input
      channels, Dout output channels, and a kernel size of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
      a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
      with a gain of 1 (Xavier initialization).

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer. For a
      linear layer it should have shape (Din, Dout); for a convolution layer it
      should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.

    if K is None:

        if relu:
            weight = (2/Din)**(1/2) * torch.randn(Din,Dout)
        else:
            weight = (1/Din)**(1/2) * torch.randn(Din,Dout)

    else:

        if relu:
            weight = (2/(Din*K*K))**(1/2) * torch.randn(Din,Dout, K, K)
        else:
            weight = (1/(Din*K*K))**(1/2) * torch.randn(Din,Dout, K, K)

    return weight

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0/ N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx


def sample_batch(X, y, num_train, batch_size):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """

    batch_idxes = np.random.randint(0, num_train, batch_size)
    
    # Get the corresponding samples and labels
    X_batch = X[batch_idxes]
    y_batch = y[batch_idxes]
    
    return X_batch, y_batch


def plot_loss_and_accuracy(epochs , loss_history , accuracy_history):
    
    # Create the figure and two subplots
    plt.figure(figsize=(12, 6))

    # Plot the loss curve
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), loss_history, label="Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)

    # Plot the accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), accuracy_history, label="Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)


    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust the top to make room for the suptitle
    plt.savefig("loss_and_accuracy_vs_epochs2.png")
    print("Graphique de la perte et de l'accuracy sauvegardé sous 'loss_and_accuracy_vs_epochs.png'")  