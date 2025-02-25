import numpy as np
from tensorflow.keras.models import model_from_json
import os

def load_model(model_json_path, weights_path):
    with open(model_json_path, 'r') as model_file:
        model_json = model_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model

def test_model(model, test_data):
    predictions = model.predict(test_data)
    return predictions

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'Data', 'Model', 'model.json')
    weights_path = os.path.join(current_dir, 'Data', 'Model', 'weights.weights.h5')

    model = load_model(model_path, weights_path)
    
    test_data = np.random.rand(1, 150, 150, 3)  
    predictions = test_model(model, test_data)
    
    print("Predictions:", predictions) 