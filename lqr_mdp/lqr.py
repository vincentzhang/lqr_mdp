import numpy as np
import scipy.linalg
import json
import logging

logger=logging.getLogger(__name__)


class MDP_LQR(object):
    """ Base class of a LQR MDP """
    
    def __init__(self, dim_x=2, dim_u=2, gamma=0.9, horizon=np.inf, sigma_w=0, std_x=1.0, max_iter=1e5, max_reward=1e5):
        """ Constructor that creates an LQR with random parameters A,B,S,R
        
        Args:
            dim_x (int): Dimension of state
            dim_u (int): Dimension of action
            gamma (float): The discount factor
            horizon (float): Horizon
            sigma_w (float): standard deviation of the transition/process noise
            std_x (float): standard deviation of init state 
            max_iter (float): max limit of the number of iterations
            max_reward (float): max limit of the sum of rewards

        Attrs:
            iter (int): algorithm iteration
            iter_sys (int): environment iteration
            sum_rewards (float): discounted reward
            total_rewards (float): undiscounted reward
            done (bool): True if terminated, otherwise False
            terminal_cost (float): cost at the terminal state
            P (np.array): cost matrix for the current policy
            P_optimal (np.array): cost matrix for the optimal policy

        """
        
        assert horizon > 0, "horizon needs to be positive or np.inf"
        assert gamma > 0.0 and gamma <= 1.0, "discounting is between 0 and 1"

        self.dim_x = dim_x
        self.dim_u = dim_u
        self.A = np.random.normal(size=(dim_x, dim_x))
        self.B = np.random.normal(size=(dim_x, dim_u))
        ## Make S and R diagonal with positive diagonal entries in (0,1.0]
        ## This is a special case but general enough in practice.
        self.S = np.diag(1.0 - np.random.random_sample(size=(dim_x)))
        self.R = np.diag(1.0 - np.random.random_sample(size=(dim_u)))
        self.horizon = horizon
        self.sigma_w = sigma_w # std of the process noise
        self.std_x = std_x # std of the init state
        self.x_thres = 1e-1 # (optional) x less then this threshold is considered terminal states
        self.x = np.random.normal(scale=self.std_x, size=dim_x) # vector instead of 1xn matrix
        self.iter = 0 # algorithm iteration
        self.iter_sys = 0 # environment iteration
        self.max_iter = max_iter
        self.max_reward = max_reward

        self.gamma_ = gamma # original gamma

        self.sum_rewards = 0.0 # for discounted reward
        self.total_rewards = 0.0 # for undiscounted reward
        self.done = False
        self.terminal_cost = 0.0
        
        self.P = None # cost matrix for the current policy
        self.P_optimal = None # cost matrix for the optimal policy

    @property
    def state(self):
        return np.array(self.x)
    
    @property
    def terminal_value(self):
        return self.terminal_cost

    def save_to_json_file(self, fn):
        """ save the env to a json file """
        raise NotImplementedError
    
    def load_from_json_file(self, fn):
        """ load the env from a json file """
        raise NotImplementedError

    def compute_cost_matrix(self, K):
        """ compute the solution P to the Lyapunov equation for the current policy K """
        raise NotImplementedError

    def compute_optimal_cost_matrix(self):
        """ compute the optimal solution P_optimal to the CARE """
        raise NotImplementedError

    def get_optimal_policy(self):
        """ get the optimal policy K """
        raise NotImplementedError
        
    def is_stable(self, K=None):
        """ check if the system is open-loop stable or closed-loop stable when the policy K is provided """ 
        raise NotImplementedError
    
    def step(self, action):
        """ Run one algorithmic timestep of the environment's dynamics
        
        Returns:
            a tuple of 
            next state (np array): 
            reward (float): immediate reward
            done (bool): if the terminal has been reached
        """
        raise NotImplementedError

    def reset_dynamic(self):
        """ reset the system dynamics parameters A and B """
        self.A = np.random.normal(size=(self.dim_x, self.dim_x))
        self.B = np.random.normal(size=(self.dim_x, self.dim_u))
        logger.info("Dynamic resetted to: ")
        logger.info("A: {}".format(self.A))
        logger.info("B: {}".format(self.B))

    def value(self):
        """ return the estimated state value """
        eta = self.gamma / (1-self.gamma) * (self.sigma_w**2)
        return self.x.T.dot(self.P).dot(self.x) + eta * np.trace(self.P)  # V(x) for policy K, positive cost
    
    def is_controllable(self):
        """ check if the system is controllable """
        # Construct the W matrix
        # W = [B|AB|A^2B| ... |A^{k}B| ... |A^{n−1}B] where n is the dim of state
        W = np.copy(self.B)
        A = np.copy(self.A)
        for _ in range(1, self.dim_x):
            W = np.hstack((W, np.matmul(A, self.B)))
            A = np.matmul(A, self.A)
        if np.linalg.matrix_rank(W) == self.dim_x:
            # check if it's full rank
            return True
        else:
            return False

    def reset(self):
        """ Reset for the new episode """
        self.x = np.random.normal(scale=self.std_x, size=self.dim_x) # reset the state x
        logger.info("Init state of mdp is reset to :\n {}".format(self.x))
        self.iter = 0
        self.iter_sys = 0
        self.sum_rewards = 0.0
        self.total_rewards = 0.0
        self.done = False
        self.terminal_cost = 0.0


class MDP_LQR_Disc(MDP_LQR):
    """ Class of a discrete-time stochastic LQR """
    def __init__(self, dim_x=2, dim_u=2, gamma=0.9, horizon=np.inf, sigma_w=0, std_x=1.0, max_iter=1e5, max_reward=1e5, fn=None):
        """ Constructor that creates a discrete-time LQR with random parameters A,B,S,R
            
            Args (specific to this child class):
                fn (string): file name for saving/loading MDP

            Attributes:
                gamma (float): effective gamma

        """
        if fn:
            self.load_from_json_file(fn)
        else:
            super(MDP_LQR_Disc, self).__init__(dim_x, dim_u, gamma, horizon, sigma_w, std_x, max_iter, max_reward)

            self.gamma = self.gamma_

            # make mdp controllable and stable
            #while (not self.is_controllable()) or (not self.is_stable()):
            while not self.is_controllable():
                logger.info("Reset the dynamic parameter since the system is not controllable")
                self.reset_dynamic()
        
        logger.info("The LQ system parameters are: ")
        logger.info("dim_x: \n{}".format(self.dim_x))
        logger.info("dim_u: \n{}".format(self.dim_u))
        logger.info("A: \n{}".format(self.A))
        logger.info("B: \n{}".format(self.B))
        logger.info("S: \n{}".format(self.S))
        logger.info("R: \n{}".format(self.R))
        logger.info("sigma_w: {}".format(self.sigma_w))

    def name(self):
        return "disc"

    def save_to_json_file(self, fn):
        """ save the env to a json file """
        data = {}
        data["dim_x"] = self.dim_x
        data["dim_u"] = self.dim_u
        data["A"] = self.A.tolist()
        data["B"] = self.B.tolist()
        data["S"] = self.S.tolist()
        data["R"] = self.R.tolist()
        data["horizon"] = self.horizon
        data["sigma_w"] = self.sigma_w
        data["std_x"] = self.std_x
        data["x_thres"] = self.x_thres
        data["max_iter"] = self.max_iter 
        data["max_reward"] = self.max_reward
        data["gamma"] = self.gamma

        with open(fn, 'w') as fp:
            json.dump(data, fp, indent=4)

    def load_from_json_file(self, fn):
        """ load the env from a json file fn"""
        with open(fn, 'r') as fp:
            data = json.load(fp)
            
        self.dim_x = data["dim_x"]
        self.dim_u = data["dim_u"]
        self.A = np.array(data["A"])
        self.B = np.array(data["B"])
        self.S = np.array(data["S"])
        self.R = np.array(data["R"])
        self.horizon = np.array(data["horizon"])
        self.sigma_w = data["sigma_w"]
        self.std_x = data["std_x"]
        self.x_thres = data["x_thres"]
        if "init_x" in data.keys():
            self.x = np.array(data["init_x"])
        else: # random init
            self.x = np.random.normal(scale=self.std_x, size=self.dim_x) # vector instead of 1xn matrix
        self.iter = 0
        self.iter_sys = 0
        self.max_iter = data["max_iter"]
        self.max_reward = data["max_reward"]

        self.sum_rewards = 0.0
        self.total_rewards = 0.0
        self.done = False
        self.terminal_cost = 0.0
        self.P = None
        self.P_optimal = None

        self.gamma_ = data.get("gamma_") # original gamma
        self.gamma = data["gamma"] # effective gamma, from discretization

        # log
        logger.info("Loaded from file {}".format(fn))

    def is_stable(self, K=None):
        """ check if policy K is stable, or check open-loop stability if K is None
        
            K is not None:
                if the eigenvalues of A+BK are all strictly less than one 
            K is None:
                if the eigenvalues of A are all strictly less than one 
        """ 
        if K is None:
            eigvals = scipy.linalg.eigvals(self.A) # open loop
        else: # closed loop
            try: 
                eigvals = scipy.linalg.eigvals(self.A + self.B.dot(K))
            except ValueError as ve:
                logger.error("An error occurred in is_stable")
                raise ValueError("An error occurred in is_stable") from ve

        # eigvalsh for symmetric matrices
        for v in eigvals:
            if abs(v) >= 1:
                return False
        return True

    def compute_cost_matrix(self, K):
        """ compute the solution P to the Lyapunov equation for the current policy K """
        try:
            A_tilde = np.sqrt(self.gamma) * (self.A + self.B.dot(K))
            self.P = scipy.linalg.solve_discrete_lyapunov(A_tilde.T, self.S+K.T.dot(self.R).dot(K))
        except ValueError as ve:
            logger.error("An error occurred in compute_cost_matrix")
            raise ValueError("An error occurred in compute_cost_matrix") from ve
        finally:
            if not np.allclose(self.P, self.P.T):
                logger.error("P is not symmetric")
                raise ValueError("P is not symmetric")

    def compute_optimal_cost_matrix(self):
        """ compute the optimal solution P_optimal to the DARE """
        try:
            A_tilde = np.sqrt(self.gamma) * self.A
            B_tilde = np.sqrt(self.gamma) * self.B
            self.P_optimal = scipy.linalg.solve_discrete_are(A_tilde, B_tilde, self.S, self.R) #a, b, q, r
        except ValueError as ve:
            logger.error("An error occurred in compute_optimal_cost_matrix")
            raise ValueError("An error occurred in compute_optimal_cost_matrix") from ve
        finally:
            if not np.allclose(self.P_optimal, self.P_optimal.T):
                logger.error("P_optimal is not symmetric")
                raise ValueError("P_optimal is not symmetric")

    def get_optimal_policy(self):
        """ get the optimal policy K """
        # first compute the solution to the DARE: P_optimal
        self.compute_optimal_cost_matrix()
        # policy K = - gamma * (R + gamma* B'PB)^-1 * B' * P * A
        # action can be computed as K * x
        return - self.gamma * np.linalg.pinv(self.R+self.gamma*self.B.T.dot(self.P_optimal).dot(self.B)).dot(self.B.T).dot(self.P_optimal).dot(self.A)

    def step(self, action):
        """ Run one algorithmic timestep of the environment's dynamics
        
            Return a tuple of the next state, reward, and if the terminal has been reached
        """
        if self.done:
            logger.info("Episode already finished! Please reset the system")
            return np.array(self.x), 0.0, True
        
        # compute the reward
        reward = self.x.T.dot(self.S).dot(self.x) + action.T.dot(self.R).dot(action)
        self.total_rewards += reward # undiscounted sum of reward
        try:
            discounted_reward = (self.gamma ** self.iter) * reward # sum of discounted reward
        except FloatingPointError: # for underflow
            discounted_reward = 0
        self.sum_rewards += discounted_reward
        # evolve the state of the system xk+1 = A*xk + B*uk + sigma_w * wk
        # self.x = self.A.dot(self.x) + self.B.dot(action) + self.sigma_w*np.random.normal(size=self.dim_x)
        self.x = self.A.dot(self.x) + self.B.dot(action) + np.random.normal(size=self.dim_x, scale=self.sigma_w)
        self.iter_sys += 1
        self.iter += 1

        # has it reached terminal
        try:
            if reward > self.max_reward or self.iter > self.max_iter or np.isnan(self.x).any():
                # system diverge
                self.done = True
            elif (abs(self.x)<=self.x_thres).all():
                # converge to below the threshold
                self.done = True
            else:
                self.done = self.iter >= self.horizon

        except Exception:
            logger.error("Something messed up in terminal state checking")


        # if it's terminal, add the remaining cost for the current/terminal state
        # two options: zero terminal cost or non-zero terminal cost
        # the terminal state value is discounted more than the immediate reward
        if self.done:
            self.terminal_cost = self.x.T.dot(self.S).dot(self.x)
            try:
                discounted_reward = self.gamma ** self.iter * self.terminal_cost # sum of discounted reward
            except FloatingPointError:
                discounted_reward = 0
            self.sum_rewards += discounted_reward
            self.total_rewards += self.terminal_cost

        # return next state, current reward, and if reached terminal
        return (np.array(self.x), reward, self.done)


class MDP_LQR_Cont(MDP_LQR):
    """ Class of a continuous-time stochastic LQR """
    def __init__(self, dim_x=2, dim_u=2, h=0.01, gamma=0.9, horizon=np.inf, sigma_w=0, std_x=1.0, max_iter=1e5, max_reward=1e5, fn=None):
        """ Constructor that creates a cts-time LQR with random parameters A,B,S,R,h
        
            Args (specific to this child class):
                h (float): Time duration between steps
                fn (string): File name for saving/loading MDP

            Attributes:
                gamma (float): effective gamma
                h_sys (float): time duration of the system
                h_ratio (int): a ratio between the time duration of algorithm and system

        """
        assert h > 0, "h needs to be positive" 
        
        if fn:
            self.load_from_json_file(fn)
        else:
            super(MDP_LQR_Cont, self).__init__(dim_x, dim_u, gamma, horizon, sigma_w, std_x, max_iter, max_reward)

            # discretization
            self.h = h
            self.gamma = gamma ** h # effective gamma, from discretization
            
            # need to move the system dynamics at a fixed fast disrretization rate, h_sys = 0.0001
            # decouple the time duration of the algorithm from that of the environment
            self.h_sys = 1e-4
            self.h_ratio = int(self.h / self.h_sys) 

            # make mdp controllable and stable
            while not self.is_controllable():
                logger.info("Reset the dynamic parameter since the system is not controllable")
                self.reset_dynamic()
    
    def name(self):
        return "cont"

    def save_to_json_file(self, fn):
        """ save the env to a json file """
        data = {}
        data["dim_x"] = self.dim_x
        data["dim_u"] = self.dim_u
        data["A"] = self.A.tolist()
        data["B"] = self.B.tolist()
        data["S"] = self.S.tolist()
        data["R"] = self.R.tolist()
        data["horizon"] = self.horizon
        data["sigma_w"] = self.sigma_w
        data["std_x"] = self.std_x
        data["x_thres"] = self.x_thres
        data["max_iter"] = self.max_iter 
        data["max_reward"] = self.max_reward
        data["gamma"] = self.gamma
        data["gamma_"] = self.gamma_
        data["h"] = self.h
        data["h_sys"] = self.h_sys

        with open(fn, 'w') as fp:
            json.dump(data, fp, indent=4)

    def load_from_json_file(self, fn):
        """ load the env from a json file """
        with open(fn, 'r') as fp:
            data = json.load(fp)
            
        self.dim_x = data["dim_x"]
        self.dim_u = data["dim_u"]
        self.A = np.array(data["A"])
        self.B = np.array(data["B"])
        self.S = np.array(data["S"])
        self.R = np.array(data["R"])
        self.horizon = np.array(data["horizon"])
        self.sigma_w = data["sigma_w"]
        self.std_x = data["std_x"]
        self.x_thres = data["x_thres"]
        if "init_x" in data.keys():
            self.x = data["init_x"]
        else: # random init
            self.x = np.random.normal(scale=self.std_x, size=self.dim_x) # vector instead of 1xn matrix
        self.iter = 0
        self.iter_sys = 0
        self.max_iter = data["max_iter"]
        self.max_reward = data["max_reward"]

        self.sum_rewards = 0.0
        self.total_rewards = 0.0
        self.done = False
        self.terminal_cost = 0.0
        self.P = None
        self.P_optimal = None

        # discretization
        self.h = data["h"]
        self.gamma_ = data.get("gamma_") # original gamma
        self.gamma = data.get("gamma", self.gamma_**self.h) # effective gamma, from discretization
        self.h_sys = data["h_sys"]
        self.h_ratio = int(self.h / self.h_sys) 

    def compute_cost_matrix(self, K):
        """ compute the solution P to the Lyapunov equation for the current policy K """
        try:
            A_tilde = self.gamma * (self.A + self.B.dot(K)) - (1-self.gamma)/(2*self.h) * np.eye(self.dim_x)
            self.P = scipy.linalg.solve_continuous_lyapunov(A_tilde.T, -(self.S+K.T.dot(self.R).dot(K)))
        except ValueError as ve:
            logger.error("An error occurred in compute_cost_matrix")
            raise ValueError("An error occurred in compute_cost_matrix") from ve
        finally:
            if not np.allclose(self.P, self.P.T):
                logger.error("P is not symmetric")
                raise ValueError("P is not symmetric")

    def compute_optimal_cost_matrix(self):
        """ compute the optimal solution P_optimal to the CARE """
        try:
            A_tilde = self.gamma * self.A - (1-self.gamma)/(2*self.h) * np.eye(self.dim_x)
            B_tilde = self.gamma * self.B
            self.P_optimal = scipy.linalg.solve_continuous_are(A_tilde, B_tilde, self.S, self.R) #a, b, q, r
        except ValueError as ve:
            logger.error("An error occurred in compute_optimal_cost_matrix")
            raise ValueError("An error occurred in compute_optimal_cost_matrix") from ve
        finally:
            if not np.allclose(self.P_optimal, self.P_optimal.T):
                logger.error("P_optimal is not symmetric")
                raise ValueError("P_optimal is not symmetric")

    def get_optimal_policy(self):
        """ get the optimal policy K """
        # first compute the solution to the CARE: P_optimal
        self.compute_optimal_cost_matrix()
        # policy K = - gamma * R^-1 * B' * P
        # action can be computed as K * x
        return - self.gamma * np.linalg.pinv(self.R).dot(self.B.T).dot(self.P_optimal)

    def is_stable(self, K=None):
        """ check if policy K is stable, or check open-loop stability if K is None
        
            K is not None:
                if the real parts of the eigenvalues of A+BK are all negative 
            K is None:
                if the real parts of the eigenvalues of A are all negative 
        """ 
        if K is None:
            eigvals = scipy.linalg.eigvals(self.A) # open loop
        else: # closed loop
            try: 
                eigvals = scipy.linalg.eigvals(self.A + self.B.dot(K))
            except ValueError as ve:
                logger.error("An error occurred in is_stable")
                raise ValueError("An error occurred in is_stable") from ve

        # eigvalsh for symmetric matrices
        for v in eigvals:
            if v.real >= 0:
                return False
        return True
    
    def step(self, action):
        """ Run one algorithmic timestep of the environment's dynamics
        
            Return a tuple of the next state, reward, and if the terminal has been reached
        """
        if self.done:
            logger.info("Episode already finished! Please reset the system")
            return np.array(self.x), 0.0, True
        
        # evolve the state of the system at the system time duration h_sys
        reward = 0.0
        for _ in range(self.h_ratio):
            # compute the reward at the system time duration
            reward += self.h_sys*(self.x.T.dot(self.S).dot(self.x) + action.T.dot(self.R).dot(action))
            # same action during this time duration h, where the system dynamics is evolved at a faster rate
            self.x += self.h_sys*(self.A.dot(self.x)+self.B.dot(action)) + self.sigma_w*np.sqrt(self.h_sys)*np.random.normal(size=self.dim_x)
            self.iter_sys += 1

        # self.x += self.h*(self.A.dot(self.x)+self.B.dot(action)) + self.sigma_w*np.sqrt(self.h)*np.random.normal(size=dim_x) # evlove dynamics based on the algorithm time duration
        # reward = self.h * (self.x.T.dot(self.S).dot(self.x) + action.T.dot(self.R).dot(action)) # reward for the algorithm time duration
        self.sum_rewards += self.gamma ** self.iter * reward # sum of discounted reward
        self.total_rewards += reward
        
        self.iter += 1
        # has it reached terminal
        try:
            if reward > self.max_reward or self.iter > self.max_iter or np.isnan(self.x).any():
                # system diverge
                self.done = True
            elif (abs(self.x)<=self.x_thres).all():
                # converge to below the threshold
                self.done = True
            else:
                self.done = self.iter >= self.horizon
        except Warning:
            logger.warning("Something messed up in terminal checking")

        # if it's terminal, add the remaining cost for the current/terminal state
        # two options: zero terminal cost or non-zero terminal cost
        # the terminal state value is discounted more than the immediate reward
        if self.done:
            self.terminal_cost = self.x.T.dot(self.S).dot(self.x)
            try:
                discounted_reward = self.gamma ** self.iter * self.terminal_cost # sum of discounted reward
            except FloatingPointError:
                discounted_reward = 0
            self.sum_rewards += discounted_reward
            self.total_rewards += self.terminal_cost

        # return next state, current reward, and if reached terminal
        return (np.array(self.x), reward, self.done)
