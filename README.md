# lqr-mdp

lqr-mdp is a simple MDP library for Linear Quadratic Regulator (LQR) problems. It provides an openai-gym like interface.

It currently supports both discrete-time and continuous-time LQR with the following cost setup:
* infinite-horizon discounted cost with or without process noise
* infinite-horizon undiscounted cost without process noise

# Dependency

* numpy
* scipy (for the ARE solver)
* json (for saving and loading)

# Usage

```python
# import modules
import numpy as np
from lqr import MDP_LQR_Disc

# Load an MDP
mdp = MDP_LQR_Disc(fn='env/mdp_disc_laplacian.json')

# Print the starting state
print("The starting state is {}".format(mdp.state))

# Get the optimal policy and run it for 10 steps
K_optimal = mdp.get_optimal_policy()
for _ in range(10):
    # take optimal action and print the state
    action = K_optimal.dot(mdp.state)
    next_state, reward, done = mdp.step(action)
    print("Iter {}: Next state: {}, cost: {}, done: {}".format(mdp.iter, next_state, reward, done))
# Print the Optimal Value Matrix P_optimal where V(x) = x'P_optimal x
print("The optimal value matrix P_optimal: \n", mdp.P_optimal)
```