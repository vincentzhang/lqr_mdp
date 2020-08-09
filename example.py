import numpy as np
from lqr_mdp import MDP_LQR_Disc

# load the discrete-time LQR instance from file
mdp = MDP_LQR_Disc(fn='env/mdp_disc_laplacian.json')

print("The starting state is {}".format(mdp.state))

# Get a random linear policy K and run it for 10 steps
K = np.random.random((mdp.dim_x, mdp.dim_u))
for _ in range(10):
    # take random actions and print the state
    action = K.dot(mdp.state)
    next_state, reward, done = mdp.step(action)
    print("Iter {}: Next state: {}, cost: {}, done: {}".format(mdp.iter, next_state, reward, done))
    if done:
        # In this case, it's either the cost is bigger than the maximum, or that the state is NaN
        print("Time to terminate")
        break
        
# Print the Value Matrix P for the current policy K where V(x) = x'Px
mdp.compute_cost_matrix(K)
print("The value matrix P: \n", mdp.P)

# Get the optimal policy and run it for 10 steps
K_optimal = mdp.get_optimal_policy()
mdp.reset()
print("The state is reset to {}".format(mdp.state))
for _ in range(10):
    # take optimal action and print the state
    action = K_optimal.dot(mdp.state)
    next_state, reward, done = mdp.step(action)
    print("Iter {}: Next state: {}, cost: {}, done: {}".format(mdp.iter, next_state, reward, done))

# Print the Optimal Value Matrix P_optimal where V(x) = x'P_optimal x
print("The optimal value matrix P_optimal: \n", mdp.P_optimal)