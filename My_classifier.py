from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Definição do modelo a se utilizar
classifficador = sequential()

# Convolução 1
classifficador.add(Conv2D(32(3,3),input_shape = (64,64,3), padding=same, activation='relu'))
# Convolução 2
classifficador.add(Conv2D(32(3,3),activation = 'relu'))
# Max pooling 1
classifficador.add(MaxPooling2D(pool_size =(2,2)))
# Dropout 1
classifficador.add(Dropout(0.25))
# Convolução 3
classifficador.add(Conv2D(32(3,3),activation = 'relu'))
# Convolução 4
classifficador.add(Conv2D(32(3,3),activation = 'relu'))
# Max pooling 2
classifficador.add(MaxPooling2D(pool_size =(2,2)))
# Dropout 2
classifficador.add(Dropout(0.25))
# flattening
classifficador.add(Flatten())
# Camada Dense(totalmente conectada)
classifficador.add(Dense(128, activation = 'relu'))
classifficador.add(Dense(64,activation = 'relu'))
classifficador.add(Dense(1,activation ='softmax'))
# Entrando na CNN de fato
classifficador.compile(
    optimizer = 'sgd',
    loss = 'categorical_crossentropy',
    metrics ='categorical_accuracy'
)
# Ajustando as imgs que entrarão na rede
from keras.preprocessing.image import ImageDataGenerator

treino_data = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=180.,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
teste_data =ImageDataGenerator(
    rescale = 1./255
)
treino_config = treino_data.flow_from_directory(
    'new_dataset/traning_set',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'categorical'
)
teste_config =  teste_data.flow_from_directory(
    'new_dataset/test_set',
    target_size = (64,64),
    bach_size = 32,
    class_mode = 'categorical'
)
# Ajustando o historico de dados
historico = classifficador.fit(
    treino_config,
    steps_per_epoch =#numero de imgs dataset de treino/bach_size VER como usa a função len(treino_config)
    epochs = 5,
    validation_data = teste_config,
    validation_steps =#numero de imgs dataset de teste/bach_size VER como usa a função len(teste_config)
)

# Salvar tudo
classifficador.save('fruit_classifier_5e.h5')

# plotando o gráfico

import matplotlib.pyplot as plt

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

plt.show()
