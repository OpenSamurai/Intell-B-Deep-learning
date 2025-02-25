import numpy as np
from keras.models import model_from_json

def load_model(model_path, weights_path):
    with open(model_path, 'r') as model_file:
        model_json = model_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

if __name__ == '__main__':
    model = load_model('Data/Model/model.json', 'Data/Model/weights.weights.h5')
    
    # Create a dummy input
    test_input = np.random.rand(1, 150, 150, 3)  # Example shape for an image input😒
    print("Testing prediction...")
    Y = model.predict(test_input)
    print("Prediction output:", Y) 
