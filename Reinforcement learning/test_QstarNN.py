import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from generate_data_for_RL import *

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
NUM_EPISODES = 500  # Define the number of episodes for training

# DQN Model
def build_model():
    model = Sequential([
        Input(shape=(NUM_RULES,)),  # Specify input shape here
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(ACTION_SPACE_SIZE, activation='linear')
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
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

        print(f"Episode {episode+1}: Total Reward = {total_reward}")


# Main execution
red_wine = pd.read_csv('red_wine.csv')
[(input_train, output_train), (input_control, output_control)] = create_input_output(red_wine)

model = build_model()
train_dqn(model, input_train, output_train)

def test_all():
    pass
