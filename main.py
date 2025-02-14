import act_func
import tensorflow as tf
import os
import numpy as np
from utils import save_model, load_model , load_and_preprocess_image 
from neural_network import NeuralNetwork



if __name__ == '__main__':
    model_path = "classification.json"
    
    if not os.path.exists(model_path):
        # Define the architecture [input, hidden1, hidden2, ..., output]
        nn = NeuralNetwork([3072, 256 , 128 , 2] , optimizer="Adam" , loss='cross_entropy_loss' ,file=model_path)

        # Training data
        #Classes à filtrer
        class_names = ["Avion", "Automobile", "Oiseau", "Chat", "Cerf", "Chien", "Grenouille", "Cheval", "Bateau", "Camion"]
        classes_to_save = ["Chien", "Cheval"] 
        class_indices = [class_names.index(cls) for cls in classes_to_save] #[5, 7]

        # Charger le dataset CIFAR-10
        (X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

        # Filtrer les données pour ne conserver que les chiens et les chevaux
        mask = np.isin(y_train, class_indices)
        X_train = X_train[mask.flatten()][:1000]/255
        y_train = y_train[mask.flatten()][:1000]

        # Convertir les labels en one-hot encoding
        y_train_one_hot = np.zeros((len(y_train), len(class_indices)))
        for i, label in enumerate(y_train.flatten()):
            y_train_one_hot[i, class_indices.index(label)] = 1

        activation_funcs = [act_func.ReLU, act_func.ReLU, act_func.softmax]  
        activation_funcs_prime = [act_func.ReLUPrime, act_func.ReLUPrime, act_func.softmaxPrime]

        # Train the network
        print(f"Optimizer : {nn.optimizer}")
        final_loss , accuracy = nn.train(X_train, y_train_one_hot ,activation_funcs , activation_funcs_prime,
                eta=1e-3, epochs=10, minibatch=200 )

        print("Modèle entraîné et sauvegardé sous", model_path)
    else:
        # Charger le modèle depuis le fichier
        nn = load_model(NeuralNetwork, model_path)
        print("Modèle chargé depuis", model_path)

    image_path = "test_input/chien_japonais.png"
    input_shape = (32, 32)  # Taille cible de l'image (doit correspondre à la taille des entrées du modèle)
    image_array = load_and_preprocess_image(image_path, input_shape)

    # Faire une prédiction
    activation_funcs = [act_func.ReLU, act_func.ReLU, act_func.softmax]  # Les mêmes fonctions d'activation que lors de l'entraînement
    prediction = nn.predict(image_array, activation_funcs)

    # Interpréter la sortie (probabilités)
    predicted_class = np.argmax(prediction)
    class_names = ["Chien", "Cheval"]  # Classes considérées lors de l'entraînement
    print(f"L'image {image_path} est classée comme : {class_names[predicted_class]}")



