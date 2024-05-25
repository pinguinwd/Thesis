
# %%
import pandas as pd
from generate_data_for_RL import *
from test_QstarNN import *
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.preprocessing.sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
# %%




def build_model(num_categories, embedding_dim, sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=num_categories, output_dim=embedding_dim, input_length=sequence_length),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)  # Output layer for regression to produce a float
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model



# %%


def plot_training_and_validation_loss(model, padded_inputs, outputs):
    history = model.fit(padded_inputs, outputs, batch_size=32, epochs=100, validation_split=0.2)
    model.summary()
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_all():
    # Load the dataset
    red_wine = pd.read_csv('D:\\PythonProjects\\Thesis\\red_wine.csv', sep=',')

    for step in range(10):
        input_sequences, output_values = create_input_output(min_data_points, CA_SIZE, red_wine)

        # Find the maximum sequence length for padding
        sequence_length = max(len(x) for x in input_sequences)

        # Padding input sequences
        padded_inputs = pad_sequences(input_sequences, padding='post')

        # Convert output values to a numpy array
        outputs = np.array(output_values)

        # Build the model
        model = build_model(num_categories, embedding_dim, sequence_length)

        # Define the early stopping criteria and model checkpointing
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

        # Train the model with early stopping and checkpointing
        model.fit(padded_inputs, outputs, epochs=100, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

        # Optionally, you can reload the best model saved during training
        best_model = load_model('best_model.keras')

        # Plot training and validation loss
        best_model.save(model_directory + '\\model_' + str(step) + '.keras')
# %%

if __name__ == '__main__':
    # Example usage:
    CA_SIZE = 25
    Steps = 13
    min_data_points = 500
    num_categories = 256  # Number of categories in your input
    embedding_dim = 10    # Size of embedding vector
    model_directory = 'D:\\PythonProjects\\Thesis-data\\Reinforcement learning\\Tensorflow_models'
    run_all()
    test_all()
