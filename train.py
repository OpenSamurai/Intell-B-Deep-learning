import os
import numpy
import tensorflow as tf
from get_dataset import get_dataset
from get_model import get_model, save_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

epochs = 100
batch_size = 5

def train_model(model, train_dataset, validation_dataset):
    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')

    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto'))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0))
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    checkpoints.append(reduce_lr)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoints.append(early_stopping)

    model.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_dataset,
        shuffle=True,                       #fitting the dataset🦾🦾
        callbacks=checkpoints
    )

    return model

def main():
    # Get dataset
    train_dataset, X_test, Y_test = get_dataset()  # Now expects 3 values 🤔
    
    # Create validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, Y_test)).batch(batch_size)
    
    # Get Model:
    model = get_model()
    
    # Train model
    model = train_model(model, train_dataset, validation_dataset)
    save_model(model)
    return model

if __name__ == '__main__':
    main()
