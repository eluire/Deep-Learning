from os import system
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Carrega o classificador
classifier = load_model('home/mateus/PycharmProjects/Dogcat/venv/catDog_2.h5')

system('clear')

    # Carrega a imagem

    test_image = image.load_img(
    'home/mateus/Convolutional_Neural_Networks/dataset/single_prediction/{}'.format(img),
    target_size=(64, 64)
    )

    # Transforma a imagem em um array
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Classifica
    res = classifier.predict(test_image)

    # Mostra o resultado
    if res[0][0] == 1:
      prediction = 'dog'
    else:
      prediction = 'cat'

    print('\tResultado da classificação para {}: {}'.format(img, prediction))
