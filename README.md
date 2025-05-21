# AI-TRAFFIC-FLOW-OPTIMIZASION
AI-TRAFFIC FLOW OPTIMIZATION
import random
import numpy as np

# Define simulation parameters
actions = ['NS_Green', 'EW_Green']  # North-South green or East-West green
state_space = [(i, j) for i in range(5) for j in range(5)]  # (NS traffic, EW traffic)
q_table = np.zeros((len(state_space), len(actions)))

# Hyperparameters
alpha = 0.1     # Learning rate
gamma = 0.6     # Discount factor
epsilon = 0.1   # Exploration rate
episodes = 1000

def get_state_index(ns, ew):
    return state_space.index((ns, ew))

def get_reward(ns, ew, action):
    if action == 'NS_Green':
        return -ew  # cars in EW wait
    else:
        return -ns  # cars in NS wait

for episode in range(episodes):
    ns = random.randint(0, 4)
    ew = random.randint(0, 4)
    state_idx = get_state_index(ns, ew)

    if random.uniform(0, 1) < epsilon:
        action_idx = random.randint(0, len(actions)-1)
    else:
        action_idx = np.argmax(q_table[state_idx])

    action = actions[action_idx]
    reward = get_reward(ns, ew, action)

    new_ns = max(0, ns - 1) if action == 'NS_Green' else ns + 1
    new_ew = max(0, ew - 1) if action == 'EW_Green' else ew + 1
    new_ns = min(new_ns, 4)
    new_ew = min(new_ew, 4)

    new_state_idx = get_state_index(new_ns, new_ew)
    old_value = q_table[state_idx, action_idx]
    next_max = np.max(q_table[new_state_idx])

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state_idx, action_idx] = new_value

# Print optimized Q-table
print("Optimized Q-Table:")
for i, state in enumerate(state_space):
    print(f"State {state}: {q_table[i]}")



