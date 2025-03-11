import os
import platform
import numpy as np
from time import sleep
from PIL import ImageGrab
from tensorflow.keras.models import model_from_json
import tensorflow as tf

from game_control import get_key, press, release, click
from predict import predict

print(tf.__version__)

def load_model():
    with open('Data/Model/model.json', 'r') as model_file:
        model = model_from_json(model_file.read())
    model.load_weights("Data/Model/weights.weights.h5")
    return model

def main():
    model = load_model()
    print('AI started!')

    while True:
        screen = np.array(ImageGrab.grab())
        action = np.argmax(predict(model, screen))
        
        if action == 1:
            key = get_key(1)
            press(key)
            sleep(0.1)
            release(key)
        elif action == 2:
            click(100, 200)

if __name__ == '__main__':
    main()
