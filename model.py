import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# Import des jeux de données
df = pd.read_csv('data/mnist_train_small.csv', header=None)
df_validation = pd.read_csv('data/mnist_test.csv', header=None)


# Création séparation des features / target
y_train = to_categorical(df[0])
X_train = df.drop([0], axis=1).values.reshape(-1,28,28)

y_test = to_categorical(df_validation[0])
X_test = df_validation.drop([0], axis=1).values.reshape(-1,28,28)
y_train.shape, X_train.shape



# Conception d'un modèle de réseau de neuronne à convolution
my_model = Sequential()

my_model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28, 28, 1)))
my_model.add(MaxPooling2D(pool_size=(2,2)))

my_model.add(Conv2D(filters=16, kernel_size=(3,3)))
my_model.add(MaxPooling2D(pool_size=(2,2)))

my_model.add(Conv2D(filters=8, kernel_size=(3,3)))
my_model.add(MaxPooling2D(pool_size=(2,2)))

my_model.add(Flatten())

my_model.add(Dense(10, activation='softmax'))

# Compilation du modèle optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrainement du modèle
my_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Exportation du modèle
my_model.save('mnist_model.h5')