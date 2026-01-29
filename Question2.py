import numpy as np

# Grid world parameters
grid_size = 5
goal = (4, 4)  # bottom-right as goal

# Actions: 0-up, 1-down, 2-left, 3-right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
n_actions = len(actions)

# Initialize Q-table: 25 states x 4 actions
Q = np.zeros((grid_size * grid_size, n_actions))  # [code_file:1]

# Hyperparameters
alpha = 0.5
gamma = 0.9
epsilon = 0.1
n_episodes = 1000  # [web:7]

def state_to_pos(s):
    return divmod(s, grid_size)

def pos_to_state(i, j):
    return i * grid_size + j

def step(state, action):  # [web:11]
    i, j = state_to_pos(state)
    di, dj = actions[action]
    ni, nj = i + di, j + dj
    if 0 <= ni < grid_size and 0 <= nj < grid_size:
        next_state = pos_to_state(ni, nj)
        reward = 10 if (ni, nj) == goal else -1
    else:
        next_state = state
        reward = -1
    done = (ni, nj) == goal
    return next_state, reward, done

# Training loop
for ep in range(n_episodes):
    state = 0  # start at (0,0)
    while True:
        # Epsilon-greedy action selection [web:11]
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done = step(state, action)
        
        # Q-learning update: Q(s,a) <- Q(s,a) + alpha*(r + gamma*maxQ(s') - Q(s,a)) [web:7][web:11]
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
        
        state = next_state
        if done:
            break

print("Learned Q-values (25 states x 4 actions):")
print(Q)  # [code_file:1]
