import os
import sys
import platform
import numpy as np
from time import sleep
from PIL import ImageGrab
from game_control import *
from predict import predict
from get_dataset import save_img
from multiprocessing import Process
from tensorflow.keras.models import model_from_json
from pynput.mouse import Listener as mouse_listener
from pynput.keyboard import Listener as key_listener
from PIL import Image

def get_screenshot():
    img = ImageGrab.grab()
    img = np.array(img)[:,:,:3] # Get first 3 channel from image as numpy array.
    img = Image.fromarray(img).resize((150, 150))
    img = np.array(img)
    return img

def save_event_keyboard(data_path, event, key):
    # First convert pynput key object to string representation
    try:
        key_str = key.char  # For normal character keys
    except AttributeError:
        key_str = key.name  # For special keys like 'shift', 'ctrl' etc.

    # Check if key is in registered keys before converting to ID
    if key_str not in get_keys():
        print(f"Key '{key_str}' not found in registered keys.")
        return
    
    # Now safely convert to key ID
    key = get_id(key_str)
    data_path = data_path + '/-1,-1,{0},{1}.png'.format(event, key)
    screenshot = get_screenshot()
    save_img(data_path, screenshot)
    return

def save_event_mouse(data_path, x, y):
    data_path = data_path + '/{0},{1},0,0.png'.format(x, y)
    screenshot = get_screenshot()
    save_img(data_path, screenshot)
    return

def listen_mouse():
    data_path = 'Data/Train_Data/Mouse'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def on_click(x, y, button, pressed):
        save_event_mouse(data_path, x, y)

    def on_scroll(x, y, dx, dy):
        pass
    
    def on_move(x, y):
        pass

    with mouse_listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()

def listen_keyboard():
    data_path = 'Data/Train_Data/Keyboard'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def on_press(key):
        save_event_keyboard(data_path, 1, key)

    def on_release(key):
        save_event_keyboard(data_path, 2, key)

    with key_listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    dataset_path = 'Data/Train_Data/'
    model_path = 'Data/Model/model.json'
    
    # Create the dataset directory if it doesn't exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Create the model directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if model file exists
    if not os.path.exists(model_path):
        # Create an empty model file or handle the error as needed
        with open(model_path, 'w') as f:
            f.write('{}')  # Writing an empty JSON object as a placeholder

    # Start to listening mouse with new process:
    Process(target=listen_mouse, args=()).start()
    listen_keyboard()
    return

if __name__ == '__main__':
    main()
