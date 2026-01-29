import numpy as np

# Grid size
rows = 5
cols = 5

# Actions: 0=up, 1=down, 2=left, 3=right
actions = ['^', 'v', '<', '>']

# Delta for moves
deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Rewards: -1 everywhere, +10 at goal (4,4), 0 at goal after update but since terminal effectively handled by policy
R = np.full((rows, cols), -1.0)
goal = (4, 4)
R[goal] = 10.0

# Value function initialize to 0
V = np.zeros((rows, cols))

# Policy initialize randomly or none
policy = np.array([[' ' for _ in range(cols)] for _ in range(rows)])

# Parameters
gamma = 0.9
theta = 1e-6  # convergence threshold

# Value iteration
while True:
    delta = 0
    for i in range(rows):
        for j in range(cols):
            if (i, j) == goal:
                continue  # terminal
            v_old = V[i, j]
            max_q = float('-inf')
            best_a = 0
            for a in range(4):
                ni = i + deltas[a][0]
                nj = j + deltas[a][1]
                if 0 <= ni < rows and 0 <= nj < cols:
                    q = R[i, j] + gamma * V[ni, nj]
                else:
                    q = R[i, j] + gamma * V[i, j]  # bump into wall, stay and -1
                if q > max_q:
                    max_q = q
                    best_a = a
            V[i, j] = max_q
            policy[i, j] = actions[best_a]
            delta = max(delta, abs(v_old - V[i, j]))
    if delta < theta:
        break

# Print optimal policy
print("Optimal Policy (5x5 grid, G at (4,4)):")
for row in policy:
    print(' '.join(row))
print("Goal ^ v < > indicate optimal actions")
