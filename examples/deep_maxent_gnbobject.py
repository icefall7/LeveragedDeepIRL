"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import irl.deep_maxent_mixdemo as deep_maxent_mixdemo
import irl.deep_maxent as deep_maxent
import irl.mdp.objectworld as objectworld
from irl.value_iteration import find_policy
from irl.value_iteration import optimal_value
from irl.value_iteration import value

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
    wind = 0.2
    trajectory_length = 8
    l1 = l2 = 0
    result_vec1 = []
    result_vec2 = []

    min_p_mixed = float('inf')
    min_p_good = float('inf')

    for _ in range(3):
        results_1 = []
        results_2 = []
        ########################## Good demo #######################################
        ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,
                                     discount, True)
        ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
        # print(ow.transition_probability.shape[0],ow.transition_probability.shape[1])
        policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                             ground_r, ow.discount, stochastic=False)


        # for trajectory in trajectories:
        # 	print("\n dude")
        # 	for traj in trajectory:
        # 		xi,yi = (ow.int_to_point(traj[0]))
        # 		print(xi,yi)
        # 		print (traj[0],traj[1],traj[2])
        # 		print("aaa")

        ########################## Bad demo #######################################
        bow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,
                                      discount, False)
        ground_r_bad = np.array([bow.reward(s) for s in range(bow.n_states)])
        # print(ow.transition_probability.shape[0],ow.transition_probability.shape[1])
        policy_bad = find_policy(bow.n_states, bow.n_actions, bow.transition_probability,
                                 ground_r_bad, bow.discount, stochastic=False)


        #########################iterate diff sample size######################################
        list1= [8,16,32,64,128,256]
        # list1 = [n_trajectories]
        for n_trajectories in list1:
            print( "\n\n\ncurrent trajs size:", n_trajectories )
            trajectories = ow.generate_trajectories(n_trajectories, trajectory_length, lambda s: policy[s])
            trajectories_bad = ow.generate_trajectories(n_trajectories, trajectory_length, lambda s: policy_bad[s])

            ########################## Mixed IRL #######################################
            feature_matrix = ow.feature_matrix(discrete=False)
            r = deep_maxent_mixdemo.irl((feature_matrix.shape[1],) + structure, feature_matrix,
                                        ow.n_actions, discount, ow.transition_probability, trajectories[0:n_trajectories//2], trajectories_bad[0:n_trajectories//2],
                                        epochs,
                                        learning_rate, l1=l1, l2=l2)

            ########################## Good IRL #######################################
            r1 = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
                                 ow.n_actions, discount, ow.transition_probability, trajectories, epochs,
                                 learning_rate, l1=l1, l2=l2)

            recover_policy = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                                         r, ow.discount, stochastic=False)

            recover_policy_good = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                                              r1, ow.discount, stochastic=False)

            diff1_p = (float)(np.sum( (recover_policy - policy) !=0 )) / len(policy) * 100
            diff2_p = (float)(np.sum((recover_policy_good - policy) != 0)) / len(policy) * 100
            print ("policy difference",diff1_p,diff2_p)
            # print ( np.sum( (recover_policy - policy) !=0 ), (float)(np.sum((recover_policy_good - policy) != 0)))

            # value_orginal = value(policy, ow.n_states, ow.transition_probability, ground_r, discount)
            value_orginal = optimal_value(ow.n_states, ow.n_actions, ow.transition_probability, ground_r,
                      discount)
            value_mixed = optimal_value(ow.n_states, ow.n_actions, ow.transition_probability, r,
                      discount)
            value_good = optimal_value(ow.n_states, ow.n_actions, ow.transition_probability, r1,
                      discount)

            diff1 = np.linalg.norm(value_mixed - value_orginal) / np.linalg.norm(value_orginal)*100
            diff2 = np.linalg.norm(value_good - value_orginal) / np.linalg.norm(value_orginal)*100
            # print(diff1)
            # print(diff2)

            results_1.append(diff1_p)
            results_2.append(diff2_p)

            # determine if its the best so far
            if (min_p_mixed >= diff1_p):
                best_mix_policy = recover_policy
                best_mix_reward = r
                best_mix_value = value_mixed
                best_mix_sample_size = n_trajectories
                min_p_mixed = diff1_p

                best_good_policy = recover_policy_good
                best_good_reward = r1
                best_good_value = value_good
                best_good_sample_size = n_trajectories

                best_ground_r = ground_r
                best_ground_policy = policy
                best_ground_value = value_orginal

            if (min_p_good >= diff2_p):
                min_p_good = diff2_p


        # print(policy)
        # print("\npolicy difference:mixed->")
        # print(diff1_p)
        # print("\npolicy difference:good only->")
        # print(diff2_p)

        # print("\n\n\n")
        # print("\nvalue difference:mixed->")
        # print(results_1)
        # print("\nvalue difference:good only->")
        # print(results_2)
        result_vec1.append(results_1)
        result_vec2.append(results_2)

    results_1_1 = np.average(result_vec1, axis=0)
    results_2_2 = np.average(result_vec2, axis=0)

    print("For mixed IRL, smallest policy error is %3.2f at sample size of %d"%(min_p_mixed, best_mix_sample_size))
    print("For Good IRL, smallest policy error is %3.2f at sample size of %d\n"%(min_p_good, best_good_sample_size))

    print("For mixed IRL, average error for size", list1, results_1_1)
    print("For Good IRL, average error for size", list1, results_2_2)

    lin1, = plt.plot(list1,results_1_1,"b-")
    lin2, = plt.plot(list1, results_2_2, "r--")
    plt.title("Error in policy")
    plt.xlabel("Sample sizes")
    plt.ylabel("Error (%)")
    plt.legend([lin1, lin2], ['Mixed Demos', 'Only good Demos'])


    # print (best_ground_policy)
    # print (best_mix_policy)
    # print (best_ground_policy - best_mix_policy)
    # print ((float)(np.sum((recover_policy - policy) != 0)))
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.pcolor(best_ground_r.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("True reward")
    plt.subplot(3, 3, 2)
    plt.pcolor(best_mix_reward.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Reward-Mixed")
    plt.subplot(3, 3, 3)
    plt.pcolor(best_good_reward.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Reward-Good")

    plt.subplot(3, 3, 4)
    plt.pcolor(best_ground_policy.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Optimal Policy")
    plt.subplot(3, 3, 5)
    plt.pcolor(best_mix_policy.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Policy-Mixed")
    plt.subplot(3, 3, 6)
    plt.pcolor(best_good_policy.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Policy-Good")

    plt.subplot(3, 3, 7)
    plt.pcolor(best_ground_value.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Expected Value")
    plt.subplot(3, 3, 8)
    plt.pcolor(best_mix_value.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Recovered Value-Mixed")
    plt.subplot(3, 3, 9)
    plt.pcolor(best_good_value.reshape((grid_size, grid_size)), cmap='jet')
    plt.colorbar()
    plt.title("Recovered Value-Good")

    plt.show()


if __name__ == '__main__':
    main(10, 0.90, 10, 3, 256, 100, 0.01, (3, 4))
## grid_size, discount, n_objects, n_colours, n_trajectories, epochs,
##         learning_rate, structure