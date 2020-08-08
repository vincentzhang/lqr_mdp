import numpy as np
from lqr import MDP_LQR_Disc

# load the discrete-time LQR instance from file
mdp = MDP_LQR_Disc(fn='env/mdp_disc_laplacian.json')

print("The starting state is {}".format(mdp.state))

# random policy
for i in range(10):
    # take random actions and print the state
    action = np.random.random(mdp.dim_u)
    next_state, reward, done = mdp.step(action)
    print("Iter {}: Next state: {}, cost: {}, done: {}".format(mdp.iter, next_state, reward, done))

# optimal policy
mdp.reset()
print("The state is reset to {}".format(mdp.state))
K = mdp.get_optimal_policy()
for i in range(10):
    # take optimal action and print the state
    action = K.dot(mdp.state)
    next_state, reward, done = mdp.step(action)
    print("Iter {}: Next state: {}, cost: {}, done: {}".format(mdp.iter, next_state, reward, done))
