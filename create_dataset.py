import os  # For file and directory operations
import sys  # For system-specific parameters and functions
import platform  # For detecting the OS
import numpy as np  # For handling numerical operations
from time import sleep  # For introducing delays
from PIL import ImageGrab  # For capturing screenshots
from game_control import *  # Importing functions for game controls (e.g., key presses, mouse actions)
from predict import predict  # Importing the prediction function
from get_dataset import save_img  # Importing function to save images for training data
from multiprocessing import Process  # For running processes in parallel
from tensorflow.keras.models import model_from_json  # To load a model from a JSON file
from pynput.mouse import Listener as mouse_listener  # For capturing mouse inputs
from pynput.keyboard import Listener as key_listener  # For capturing keyboard inputs
from PIL import Image  # For image processing

def get_screenshot():
    """
    Captures a screenshot, converts it to an RGB image (3 channels), resizes it to (150x150),
    and returns it as a NumPy array.
    """
    img = ImageGrab.grab()  # Capture screen
    img = np.array(img)[:, :, :3]  # Keep only the first 3 color channels (RGB)
    img = Image.fromarray(img).resize((150, 150))  # Resize the image to 150x150
    img = np.array(img)  # Convert back to NumPy array
    return img

def save_event_keyboard(data_path, event, key):
    """
    Saves a keyboard event along with a screenshot.
    
    Parameters:
    - data_path: The directory where the data should be saved.
    - event: 1 for key press, 2 for key release.
    - key: The key that was pressed or released.
    """
    try:
        key_str = key.char  # Convert normal character keys to string
    except AttributeError:
        key_str = key.name  # Convert special keys (e.g., 'shift', 'ctrl') to string

    # Check if the key is registered before converting to an ID
    if key_str not in get_keys():
        print(f"Key '{key_str}' not found in registered keys.")
        return
    
    # Convert key string to key ID
    key = get_id(key_str)
    data_path = data_path + '/-1,-1,{0},{1}.png'.format(event, key)  # Format file name with event and key ID
    screenshot = get_screenshot()  # Capture the current screen
    save_img(data_path, screenshot)  # Save the screenshot with metadata
    return

def save_event_mouse(data_path, x, y):
    """
    Saves a mouse event (click) along with a screenshot.

    Parameters:
    - data_path: The directory where the data should be saved.
    - x, y: The coordinates of the mouse event.
    """
    data_path = data_path + '/{0},{1},0,0.png'.format(x, y)  # Format file name with mouse coordinates
    screenshot = get_screenshot()  # Capture the current screen
    save_img(data_path, screenshot)  # Save the screenshot with metadata
    return

def listen_mouse():
    """
    Listens for mouse events and saves the clicks with screenshots.
    """
    data_path = 'Data/Train_Data/Mouse'
    
    # Create the directory if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def on_click(x, y, button, pressed):
        save_event_mouse(data_path, x, y)  # Save click event
    
    def on_scroll(x, y, dx, dy):
        pass  # Scroll events are ignored
    
    def on_move(x, y):
        pass  # Mouse movement events are ignored

    # Start listening for mouse events
    with mouse_listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()

def listen_keyboard():
    """
    Listens for keyboard events and saves the key presses/releases with screenshots.
    """
    data_path = 'Data/Train_Data/Keyboard'
    
    # Create the directory if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    def on_press(key):
        save_event_keyboard(data_path, 1, key)  # Save key press event

    def on_release(key):
        save_event_keyboard(data_path, 2, key)  # Save key release event

    # Start listening for keyboard events
    with key_listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    """
    Main function to initialize directories, check model file, and start event listeners.
    """
    dataset_path = 'Data/Train_Data/'
    model_path = 'Data/Model/model.json'
    
    # Ensure the dataset directory exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Ensure the model directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if the model file exists
    if not os.path.exists(model_path):
        # Create an empty model file or handle missing model scenario
        with open(model_path, 'w') as f:
            f.write('{}')  # Write an empty JSON object as a placeholder

    # Start listening to mouse inputs in a separate process
    Process(target=listen_mouse, args=()).start()

    # Start listening to keyboard inputs in the main process
    listen_keyboard()
    return

# Execute the main function if the script is run directly
if __name__ == '__main__':
    main()
