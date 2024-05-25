import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses most TensorFlow logs unless they are errors
tf.get_logger().setLevel('ERROR')  # Correct method to set the logger level  # Set TensorFlow logger to only display error messages

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from generate_data_for_RL import *

import uuid
from tensorflow.keras.models import save_model

# Constants and Hyperparameters
NUM_RULES = 25
ACTION_SPACE_SIZE = NUM_RULES  # Assuming one action per rule
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MAX_EPSILON = 1.0
BATCH_SIZE = 32
MAX_STEPS_PER_EPLODE = 10
NUM_EPISODES = 500  # Define the number of episodes for training

# DQN Model
def build_model():
    model = Sequential([
        Input(shape=(NUM_RULES,)),  # Specify input shape here
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(ACTION_SPACE_SIZE, activation='linear')
    ])
    model.compile(loss='mse', optimizer=SGD(learning_rate=LEARNING_RATE))
    return model

# Environment Simulation
def fun_step(state, action, input, output):
    new_state = state.copy()
    # Increment action at the index with wrap-around from 255 back to 0
    new_state[action] = (state[action] + 1) % 256  
    reward = calculate_performance(new_state, input, output) - calculate_performance(state, input, output)  # Mock function to calculate reward
    return new_state, reward

# Training Loop
def train_dqn(model, input, output):
    epsilon = MAX_EPSILON
    memory = []  # For experience replay
    rewards = []

    for episode in range(NUM_EPISODES):
        state = np.random.randint(0, 256, NUM_RULES)  # Random initial state
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPLODE):
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SPACE_SIZE - 1)  # Explore action space
            else:
                action = np.argmax(model.predict(state.reshape(1, -1)))  # Exploit learned values
            
            new_state, reward = fun_step(state, action, input, output)
            total_reward += reward

            # Store experience in memory
            memory.append((state, action, reward, new_state))

            # Minibatch training from memory
            if len(memory) > BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)
                for mem_state, mem_action, mem_reward, mem_new_state in minibatch:
                    target = mem_reward + DISCOUNT_FACTOR * np.max(model.predict(mem_new_state.reshape(1, -1)))
                    target_vec = model.predict(mem_state.reshape(1, -1))[0]
                    target_vec[mem_action] = target
                    model.fit(mem_state.reshape(1, -1), target_vec.reshape(1, -1), epochs=1, verbose=0)

            state = new_state

        # Update epsilon
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    return model, rewards

def test_all():
    pass

def save_model_and_rewards(model, rewards, data_directory):
    # Generate a unique ID for the new directory
    unique_id = str(uuid.uuid4())
    dir_path = os.path.join(data_directory, unique_id)
    
    # Create the directory
    os.makedirs(dir_path, exist_ok=True)
    
    # Save the Keras model
    model_path = os.path.join(dir_path, 'model.keras')
    save_model(model, model_path)
    
    # Save the rewards to a CSV file
    rewards_path = os.path.join(dir_path, 'rewards.csv')
    rewards_df = pd.DataFrame(rewards, columns=['Reward'])
    rewards_df.to_csv(rewards_path, index=False)
    
    print(f"Model and rewards saved in directory {dir_path}")

def main():
    # Main execution
    red_wine = pd.read_csv('red_wine.csv')
    [(input_train, output_train), (input_control, output_control)] = create_input_output(red_wine)

    model = build_model()
    model, rewards = train_dqn(model, input_train, output_train)

    data_directory = 'D:\\PythonProjects\\Thesis-data\\Reinforcement learning'

    save_model_and_rewards(model, rewards, data_directory)

if __name__ == '__main__':
    main()