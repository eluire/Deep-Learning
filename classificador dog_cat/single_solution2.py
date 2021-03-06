from os import system
import numpy as np
from keras.preprocessing import image
from keras.models import load_model


# Carrega o classificador
classifier = load_model('/home/mateus/catDog_2.h5')

# Load da img
test_image = image.load_img(
    '/home/mateus/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg',
    target_size = (64, 64)
)
# Transformando img em uma array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# Classifica
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print (prediction)
    
