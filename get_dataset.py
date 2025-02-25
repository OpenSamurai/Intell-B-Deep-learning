import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from imageio import imread, imwrite
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_img(data_path):
    # Load the image using imageio
    img = imread(data_path)
    return img

def resize_img(img, size):
    img = np.array(img)  # Ensure img is a numpy array
    return np.resize(img, size)  # Use np.resize instead of img.resize

def save_img(data_path, img):
    # Save the image using imageio
    imwrite(data_path, img)

def get_img(data_path):
    # Getting image array from path:
    img = load_img(data_path)
    img = resize_img(img, (150, 150))
    return img

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
        
        # Ensure existing data has correct shape
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)
            X = np.repeat(X, 3, axis=-1)
            
    except:
        labels = os.listdir(dataset_path)  # Getting labels
        X = []
        Y = []
        count_categori = [-1, '']  # For encode labels
        for label in labels:
            datas_path = dataset_path + '/' + label
            for data in os.listdir(datas_path):
                img = get_img(datas_path + '/' + data)
                X.append(img)
                # For encode labels:
                if data != count_categori[1]:
                    count_categori[0] += 1
                    count_categori[1] = data.split(',')
                Y.append(count_categori[0])
        # Create dataset:
        X = np.array(X).astype('float32') / 255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_classes=60)

        # Force 4D shape with 3 channels
        if X.ndim == 3:
            X = X.reshape(-1, 150, 150, 1)
        X = np.repeat(X, 3, axis=-1)  # Now shape (N, 150, 150, 3)

        # Save processed data
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)

    # Final validation before return
    if X.ndim != 4:
        X = np.expand_dims(X, axis=-1)
        X = np.repeat(X, 3, axis=-1)

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    batch_size = 32  # Define the batch size
    return datagen.flow(X, Y, batch_size=batch_size), X_test, Y_test
