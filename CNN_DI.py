import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Hola Vamos a clasificar imágenes de dientes impactados o no.

# Recorremos las carpetas de prueba 
folders_train = ["/Users/kevin22/Desktop/DICNN/Train", "/Users/kevin22/Desktop/Train_Normal", "/Users/kevin22/Desktop/Train_Mandibula"]
labels_train = [1, 0, 0]
X_train, y_train = [], []

for folder, label in zip(folders_train, labels_train):
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            X_train.append(img)
            y_train.append(label)

# Convertimos las listas a arrays y normalizamos los valores de pixeles
X_train, y_train = np.array(X_train) / 255.0, np.array(y_train)

# Dividir los datos de entramiento en conjuntos de entramiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Cargar y preprocesar datos de entrenamiento
folders_test = ["/Users/kevin22/Desktop/DICNN/Test", "/Users/kevin22/Desktop/Test_Normal", "/Users/kevin22/Desktop/Test_Mandibula"]
labels_test = [1, 0, 0]
X_test, y_test = [], []

for folder, label in zip(folders_test, labels_test):
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            X_test.append(img)
            y_test.append(label)

X_test, y_test = np.array(X_test) / 255.0, np.array(y_test)

# Definir y compilar el modelo
model = models.Sequential()
model.add(layers.Conv2D(filters=50, kernel_size=(4, 4), input_shape=(28, 28, 1), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))  # Clasificación binaria, así que usamos la activación 'sigmoid'

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Hora de entrenar el modelo
history = model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=50, validation_data=(X_val.reshape(-1, 28, 28, 1), y_val))

# Evaluamos el modelo en el conjunto de prueba 
y_pred = (model.predict(X_test.reshape(-1, 28, 28, 1)) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)

# Mostramos la matriz de confusión
print("Matriz de confusion:")
print(conf_matrix)

# Guardamos el modelo entrenado 
model.save("teeth_classification_model.h5")

# Listo, ¡modelo entrenado y guardado!