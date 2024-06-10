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

from tensorflow.keras.optimizers import Adam


# Constants and Hyperparameters
NUM_RULES = 25
ACTION_SPACE_SIZE = NUM_RULES  # Assuming one action per rule
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MAX_EPSILON = 1.0
BATCH_SIZE = 32
MAX_STEPS_PER_EPLODE = 100
NUM_EPISODES = 100  # Define the number of episodes for training

# DQN Model
def build_model():
    model = Sequential([
        Input(shape=(NUM_RULES,)),  # Specify input shape here
        Dense(14, activation='relu'),
        Dense(14, activation='relu'),
        Dense(14, activation='relu'),
        Dense(ACTION_SPACE_SIZE, activation='linear')
    ])
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(loss='mse', optimizer=optimizer)
    return model

# Environment Simulation
def fun_step(state, action, input, output):
    new_state = state.copy()
    # Increment action at the index with wrap-around from 255 back to 0
    new_state[action] = (state[action] + 1) % 256  
    reward = calculate_performance(new_state, input, output) - calculate_performance(state, input, output)  # Mock function to calculate reward
    return new_state, reward

# Training Loop
def train_dqn(model, input, output, data_directory):
    memory = []  # For experience replay
    rewards = []

    for episode in range(NUM_EPISODES):
        epsilon = MAX_EPSILON
        state = np.random.randint(0, 256, NUM_RULES)  # Random initial state
        total_reward = []

        for step in range(MAX_STEPS_PER_EPLODE):
            if random.random() < epsilon:
                action = random.randint(0, ACTION_SPACE_SIZE - 1)  # Explore action space
            else:
                action = np.argmax(model.predict(state.reshape(1, -1)))  # Exploit learned values
            
            new_state, reward = fun_step(state, action, input, output)
            total_reward.append(reward)

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

        rewards.append(sum(total_reward))

        update_rewards_and_model(model, rewards, data_directory, episode, total_reward)

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    return model, rewards

def test_all():
    pass

def update_rewards_and_model(model, rewards, data_model_directory, episode, total_reward):
    # Define the rewards file path
    rewards_path = os.path.join(data_model_directory, 'running_rewards.csv')
    episode_rewards_path = os.path.join(data_model_directory, 'running_rewards_' + str(episode) + '.csv')
    reward = rewards[-1]

    # Check if the file exists
    if os.path.exists(rewards_path):
        # If the file exists, load the existing data and append the new reward
        rewards_df = pd.read_csv(rewards_path)
        new_row = pd.DataFrame([reward], columns=['Reward'])
        rewards_df = pd.concat([rewards_df, new_row], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame
        os.makedirs(data_model_directory, exist_ok=True)
        rewards_df = pd.DataFrame([reward], columns=['Reward'])
    
    # Save the updated DataFrame to CSV
    rewards_df.to_csv(rewards_path, index=False)
    
    # If the file does not exist, create a new DataFrame
    rewards_episode_df = pd.DataFrame([[x] for x in total_reward], columns=['Reward'])
    rewards_episode_df.to_csv(episode_rewards_path, index=False)

    # Save the model if it's the last call
    # Save the Keras model
    if episode == (NUM_EPISODES - 1):
        model_path = os.path.join(data_model_directory, 'model.keras')
        save_model(model, model_path)

    print(f"Model saved in directory {data_model_directory} and rewards updated in {rewards_path}")


def main():
    # Main execution
    red_wine = pd.read_csv('red_wine.csv')
    identifier = str(uuid.uuid4())
    data_directory = 'D:\\PythonProjects\\Thesis-data\\Reinforcement learning\\' + identifier + '\\'
    [(input_train, output_train), (input_control, output_control)] = create_input_output(red_wine)

    model = build_model()
    model, rewards = train_dqn(model, input_train, output_train, data_directory)


if __name__ == '__main__':
    main()