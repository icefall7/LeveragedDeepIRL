"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.deep_maxent as deep_maxent
import irl.mdp.objectworld as objectworld
from irl.value_iteration import find_policy

def main(grid_size, discount, n_objects, n_colours, n_trajectories, epochs,
		 learning_rate, structure):
	"""
	Run deep maximum entropy inverse reinforcement learning on the objectworld
	MDP.

	Plots the reward function.

	grid_size: Grid size. int.
	discount: MDP discount factor. float.
	n_objects: Number of objects. int.
	n_colours: Number of colours. int.
	n_trajectories: Number of sampled trajectories. int.
	epochs: Gradient descent iterations. int.
	learning_rate: Gradient descent learning rate. float.
	structure: Neural network structure. Tuple of hidden layer dimensions, e.g.,
		() is no neural network (linear maximum entropy) and (3, 4) is two
		hidden layers with dimensions 3 and 4.
	"""

	wind = 0.3
	trajectory_length = 8
	l1 = l2 = 0

	ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,
								 discount,True)
	ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
	# print(ow.transition_probability.shape[0],ow.transition_probability.shape[1])
	policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
						 ground_r, ow.discount, stochastic=False)
	print("policy->\n")
	print(policy)
	trajectories = ow.generate_trajectories(n_trajectories,
											trajectory_length,
											lambda s: policy[s])

	# for trajectory in trajectories:
	# 	print("\n dude")
	# 	for traj in trajectory:
	# 		xi,yi = (ow.int_to_point(traj[0]))
	# 		print(xi,yi)
	# 		print (traj[0],traj[1],traj[2])
	# 		print("aaa")

	feature_matrix = ow.feature_matrix(discrete=False)
	r = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
		ow.n_actions, discount, ow.transition_probability, trajectories, epochs,
		learning_rate, l1=l1, l2=l2)

	recover_policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
								 r, ow.discount, stochastic=False)
	plt.subplot(2, 2, 1)
	plt.pcolor(ground_r.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Groundtruth reward")
	plt.subplot(2, 2, 2)
	plt.pcolor(r.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Recovered reward")

	plt.subplot(2, 2, 3)
	plt.pcolor(policy.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Optimal Policy")
	plt.subplot(2, 2, 4)
	plt.pcolor(recover_policy.reshape((grid_size, grid_size)))
	plt.colorbar()
	plt.title("Recovered Policy")
	plt.show()

if __name__ == '__main__':
	main(10, 0.95, 15, 2, 10, 10, 0.01, (3, 4))
