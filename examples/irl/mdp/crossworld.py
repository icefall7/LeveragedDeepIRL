"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn
from . import human_model
from . import human_model1


class Crossworld(object):
    """
	Gridworld MDP.
	"""

    def __init__(self, discount, wind=0.2, behavegood=True):
        """
		grid_size: Grid size. int.
		wind: Chance of moving randomly. float.
		discount: MDP discount. float.
		-> Gridworld
		"""

        self.behave = behavegood
        self.actions = (0, 1, 2)

        self.n_actions = len(self.actions)
        self.grid_size = 10;  ##### assume 10 steps
        self.grid_size_h = 10;  # 10 grids for predestrain position
        self.grid_size_v = 3;  # 3 grids for predestrain velocity

        self.n_states = self.grid_size * self.grid_size_h * self.grid_size_v;
        self.wind = wind
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
		Get the feature vector associated with a state integer.

		i: State int.
		feature_map: Which feature map to use (default ident). String in {ident,
			coord, proxi}.
		-> Feature vector.
		"""

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
		Get the feature matrix for this gridworld.

		feature_map: Which feature map to use (default ident). String in {ident,
			coord, proxi}.
		-> NumPy array with shape (n_states, d_states).
		"""

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
	  Convert a state int into the corresponding coordinate.

	  i: State int.
	  -> (x, y, v) int tuple.
	  (000)->(001)->(011)->(111)
	  """

        tempx = i // (self.grid_size_h * self.grid_size_v);
        tempy = (i % (self.grid_size_h * self.grid_size_v)) // self.grid_size_v;
        temyz = (i % (self.grid_size_h * self.grid_size_v)) % self.grid_size_v;
        return (tempx, tempy, temyz)

    def point_to_int(self, p):
        """
		Convert a coordinate into the corresponding state int.

		p: (x, y, v) tuple.
		-> State int.
		"""

        return p[0] * self.grid_size_h * self.grid_size_v + p[1] * self.grid_size_v + p[2]

    def neighbouring(self, i, k):
        """
		Get whether two points neighbour each other. Also returns true if they
		are the same point.

		
		-> bool.
		"""
        xi, pi, vi = self.int_to_point(i)
        xk, pk, vk = self.int_to_point(k)

        if xk >= xi and xk - xi <= 3 and (
                (pi + vi) == pk or (pi + vi > self.grid_size_h - 1 and pk == self.grid_size_h - 1)):
            return True
        return False

    def _transition_probability(self, i, j, k):
        """
		Get the probability of transitioning from state i to state k given 
		action j.

		i: State int.
		j: Action int.
		k: State int.
		-> p(s_k | s_i, a_j)
		"""

        xi, pi, vi = self.int_to_point(i)
        yj = self.actions[j]
        xk, pk, vk = self.int_to_point(k)

        if not self.neighbouring(i, k):
            return 0.0

        accurate_prob = 0.8   ## self.wind is default as 0.2
        uncertain_0 = 1 - 0.8
        uncertain_1 = uncertain_0 / 2

        human_actions = 3

        if (xi == self.grid_size - 1 or xi + yj > self.grid_size - 1) and xk == self.grid_size - 1:
            return 1.0 / human_actions
        if xi + yj == self.grid_size - 1:
            if xk == self.grid_size - 1:
                return accurate_prob / human_actions
            elif xk == self.grid_size - 2:
                return uncertain_0 / human_actions

        if yj == 0:
            if xk == xi:
                return accurate_prob / human_actions
            elif xk - xi == 1:
                return uncertain_0 / human_actions
        elif yj == 1:
            if xk == xi:
                return uncertain_1 / human_actions
            elif xk - xi == 1:
                return accurate_prob / human_actions
            elif xk - xi == 2:
                return uncertain_1 / human_actions
        elif yj == 2:
            if xk - xi == 1:
                return uncertain_1 / human_actions
            elif xk - xi == 2:
                return accurate_prob / human_actions
            elif xk - xi == 3:
                return uncertain_1 / human_actions

        return 0.0

    # def reward(self, action_int, nextstate_int):
    # 	"""
    # 	Reward for being in state state_int.

    # 	state_int: State integer. int.
    # 	-> Reward.
    # 	"""
    # 	xi, pi, vi = self.int_to_point(nextstate_int)
    # 	if xi==(len(self.grid_size)-2) and (pi ==len(size)//2 or pi ==len(size)//2+1):
    # 		# if hit people, get a negative reward
    # 		if action_int==0:
    # 			return -51.0;
    # 		return -50.0
    # 	elif xi = len(self.grid_size)-1
    # 		# if reach the end, get a positive reward
    # 		if action_int==0:
    # 			return 4.0
    # 		return 5.0
    # 	if action_int==0:
    # 		return -1.0;

    # 	return 0.0

    def reward(self, nextstate_int):
        """
		Reward for being in state state_int.

		state_int: State integer. int.
		-> Reward.
		"""
        if self.behave:
            xi, pi, vi = self.int_to_point(nextstate_int)
            if xi == (self.grid_size - 3) and (pi == self.grid_size_h // 2 or pi == self.grid_size_h // 2 + 1):
                return -10
            if xi == (self.grid_size - 2) and (pi == self.grid_size_h // 2 or pi == self.grid_size_h // 2 + 1):
                # if hit people, get a negative reward
                return -20
            if xi == (self.grid_size - 2) and (pi == self.grid_size_h // 2 - 1):
                # if hit people, get a negative reward
                return -10

            elif xi == self.grid_size - 1:
                # if reach the end, get a positive reward
                return 5
            return 0
        else:
            xi, pi, vi = self.int_to_point(nextstate_int)
            if xi == (self.grid_size - 3) and (pi == self.grid_size_h // 2 or pi == self.grid_size_h // 2 + 1):
                return 10
            if xi == (self.grid_size - 2) and (pi == self.grid_size_h // 2 or pi == self.grid_size_h // 2 + 1):
                # if hit people, get a negative reward
                return 20
            if xi == (self.grid_size - 2) and (pi == self.grid_size_h // 2 - 1):
                # if hit people, get a negative reward
                return 10

            elif xi == self.grid_size - 1:
                # if reach the end, get a positive reward
                return -5
            return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
		Calculate the average total reward obtained by following a given policy
		over n_paths paths.

		policy: Map from state integers to action integers.
		n_trajectories: Number of trajectories. int.
		trajectory_length: Length of an episode. int.
		-> Average reward, standard deviation.
		"""

        trajectories = self.generate_trajectories(n_trajectories,
                                                  trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
		The optimal policy for this gridworld.

		state_int: What state we are in. int.
		-> Action int.
		"""

        sx, sy, sv = self.int_to_point(state_int)
        if sx == 8 and (sy == 4 or sy == 5):
            return 0
        else:
            return rn.randint(1, 3)

    def optimal_policy_deterministic(self, state_int):
        raise NotImplementedError(
            "Optimal policy is not implemented for crossworld.")

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                              random_start=False):
        """
		Generate n_trajectories trajectories with length trajectory_length,
		following the given policy.

		n_trajectories: Number of trajectories. int.
		trajectory_length: Length of an episode. int.
		policy: Map from state integers to action integers.
		random_start: Whether to start randomly (default False). bool.
		-> [[(state int, action int, reward float)]]
		"""

        trajectories = []

        for _ in range(n_trajectories):
            if random_start:

                sx = rn.randint(self.grid_size)
                self.h1 = human_model.Human(rn.randint(0, 10), 0)
                sy, sv = self.h1.next_state()
            else:
                ### call human class
                if rn.random() < 0.5:
                    self.h1 = human_model.Human()
                else:
                    self.h1 = human_model.Human()

                sy, sv = self.h1.next_state()
                # print("position:",pos,"speed:",speed)
                sx = 0;

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 3)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy, sv)))]
                # print(action)
                if (0 <= sx + action < self.grid_size):
                    next_sx = sx + action
                    next_sy, next_sv = self.h1.next_state()
                # next_sy = sy
                # next_sv = sv
                else:
                    next_sx = sx
                    next_sy, next_sv = self.h1.next_state()

                state_int = self.point_to_int((sx, sy, sv))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy, next_sv))
                reward = self.reward(next_state_int)
                trajectory.append((state_int, action_int, reward))
                # print(next_sx,next_sy,next_sv)
                sx = next_sx
                sy = next_sy
                sv = next_sv

            trajectories.append(trajectory)
        # print("traj->\n")
        # print (sx)
        # print(np.array(trajectories)[:, 0, 0])


        ########################import real data#################

        # data = np.loadtxt("pass_disc_break_variables.txt", skiprows=0)
        # data = data.astype(np.int64)



        return np.array(trajectories)
