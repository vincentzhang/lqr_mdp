# lqr_mdp

lqr_mdp is a simple MDP library for Linear Quadratic Regulator (LQR) problems. It provides an openai-gym like interface.

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
from lqr_mdp import MDP_LQR_Disc

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

# Print the Optimal Value Matrix P_optimal where V*(x) = x'P_optimal x
print("The optimal value matrix P_optimal: \n", mdp.P_optimal)
```

The output:
```
The starting state is [0. 0. 0.]
Iter 1: Next state: [-1.09707716  0.20329638 -0.60040983], cost: 0.0, done: False
Iter 2: Next state: [-0.58102324 -2.09573186 -2.31055448], cost: 0.0022527353602335466, done: False
Iter 3: Next state: [ 0.93529571 -3.32400709 -3.63205045], cost: 0.018014830818447475, done: False
Iter 4: Next state: [ 0.79326478 -3.10721243 -3.83886928], cost: 0.04091432000439864, done: False
Iter 5: Next state: [ 1.20219503 -2.49148956 -5.29910714], cost: 0.04096729477720143, done: False
Iter 6: Next state: [-0.54689641 -1.18115562 -5.96150239], cost: 0.05600291416429695, done: False
Iter 7: Next state: [ 1.4534063  -0.95034838 -5.84326229], cost: 0.058056078269238906, done: False
Iter 8: Next state: [ 2.16579563  0.56893895 -5.99391278], cost: 0.05438967255191356, done: False
Iter 9: Next state: [ 0.65951193  0.67922153 -5.66818587], cost: 0.05602163020888899, done: False
Iter 10: Next state: [ 1.03281106  0.46775703 -5.63656365], cost: 0.04544265065038751, done: False
The optimal value matrix P_optimal: 
 [[0.02147721 0.00597032 0.00117097]
 [0.00597032 0.02264818 0.00597032]
 [0.00117097 0.00597032 0.02147721]]
```