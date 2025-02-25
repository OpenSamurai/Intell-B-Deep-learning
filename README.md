# Playing with Artificial Intelligence

## Running the AI Model
If you have already trained the artificial intelligence model, follow these steps to play your desired game:

1. Open the game.
2. Run the following command in the terminal:
   ```sh
   python3 ai.py
   ```

## Creating a Training Dataset
To create a dataset for training the AI model, follow these steps:

1. Run the dataset creation script:
   ```sh
   python3 create_dataset.py
   ```
2. Play your desired game while the script is running.
3. Stop the dataset creation process by pressing **Ctrl + C** in the terminal.

## Model Training
To train the AI model, use the following command:
```sh
python3 train.py
```

## Using TensorBoard for Visualization
To monitor the training progress with TensorBoard, execute:
```sh
tensorboard --logdir=Data/Checkpoints/logs
```

## Important Notes
- This implementation has been tested with **Python version 3.6.0**.
- Install the required dependencies using:
  ```sh
  sudo pip3 install -r requirements.txt
  ```

