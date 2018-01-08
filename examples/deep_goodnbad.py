"""
Run maximum entropy inverse reinforcement learning on chicken crossing game.

Yiqian Gan
ganyq@umich.edu
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.deep_maxent_mixdemo as deep_maxent_mixdemo
from   irl.value_iteration import find_policy
import irl.deep_maxent as deep_maxent
import irl.mdp.crossworld as crossworld  ##### change


def main(discount, n_trajectories, epochs,
         learning_rate, structure):
    """
    Run deep maximum entropy inverse reinforcement learning on the chicken crossing game
    MDP.

    Plots the reward function.

    map: is set as constant with 15 by 1 grid.
    observations: is set by random generator, human crossing, human could have normal
    Goal: car/chicken cross the road without



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
    l1 = l2 = 0.0
    grid_size = 10
    grid_size_h = 10
    grid_size_v = 3
    ########################Good demo###########################
    xw = crossworld.Crossworld(discount, wind, True)  # good demonstrations
    # tp = xw.transition_probability
    # temp = xw.transition_probability.sum(axis=2)
    # for ii in range(xw.n_states):
    #     for jj in range(xw.n_actions):
    #         print (np.isclose(np.sum(tp[ii,jj,:]),1))

    ground_r = np.array([xw.reward(s) for s in range(xw.n_states)])
    # print(ground_r)
    # print(xw.transition_probability.shape[0],xw.transition_probability.shape[1])
    policy = find_policy(xw.n_states, xw.n_actions, xw.transition_probability,
                         ground_r, xw.discount, stochastic=False)
    print(policy)
    # policy is wrong now

    trajectories = xw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            lambda s: policy[s], True)
    # trajectories = xw.generate_trajectories(n_trajectories,
    #                                         trajectory_length,
    #                                         xw.optimal_policy,True)
    # for trajectory in trajectories:
    #     print("\n dude")
    #     for traj in trajectory:
    #         xi,yi,vi = (xw.int_to_point(traj[0]))
    #         print(xi,yi,vi)
    #         print (traj[0],traj[1],traj[2])
    #         print("aaa")
    feature_matrix = xw.feature_matrix("indet")






    ########################bad demo###########################


    bxw = crossworld.Crossworld(discount, wind, False)  # bad demonstrations

    ground_r_bad = np.array([bxw.reward(s) for s in range(bxw.n_states)])
    policy_bad = find_policy(bxw.n_states, bxw.n_actions, bxw.transition_probability,
                         ground_r_bad, bxw.discount, stochastic=False)
    print(policy_bad)
    trajectories_bad = bxw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                             lambda s: policy_bad[s], True)
    ########################IRL###########################
    r = deep_maxent_mixdemo.irl((feature_matrix.shape[1],) + structure, feature_matrix,
                        xw.n_actions, discount, xw.transition_probability, trajectories[0:n_trajectories], trajectories_bad[0:n_trajectories],epochs,
                        learning_rate, l1=l1, l2=l2)


    ##########################good IRL#######################################
    r1 = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
                         xw.n_actions, discount, xw.transition_probability, trajectories, epochs,
                         learning_rate, l1=l1, l2=l2)


    recover_policy = find_policy(xw.n_states, xw.n_actions, xw.transition_probability,
                                 r, xw.discount, stochastic=False)

    recover_policy_good = find_policy(xw.n_states, xw.n_actions, xw.transition_probability,
                                 r1, xw.discount, stochastic=False)
    plt.subplot(2, 3, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size_h, grid_size_v))[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(2, 3, 2)
    plt.pcolor(r.reshape((grid_size, grid_size_h, grid_size_v))[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.title("Recovered reward-Mixed Demo")
    plt.subplot(2, 3, 3)
    plt.pcolor(r1.reshape((grid_size, grid_size_h, grid_size_v))[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.title("Recovered reward-Good Demo")



    plt.subplot(2, 3, 4)
    plt.pcolor(policy.reshape((grid_size, grid_size_h, grid_size_v))[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.title("Optimal Policy")
    plt.subplot(2, 3, 5)
    plt.pcolor(recover_policy.reshape((grid_size, grid_size_h, grid_size_v))[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.title("Recovered Policy-Mixed Demo")
    plt.subplot(2, 3, 6)
    plt.pcolor(recover_policy_good.reshape((grid_size, grid_size_h, grid_size_v))[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.title("Recovered Policy-Mixed Demo")

    print(policy)
    print("\n policy difference:mixed->")
    print((float)(np.sum((recover_policy - policy) != 0)) / len(policy) * 100)
    print("\n policy difference:good only->")
    print((float)(np.sum((recover_policy_good - policy) != 0)) / len(policy) * 100)

    plt.show()


if __name__ == '__main__':
    # (discount, n_trajectories, epochs, learning_rate, structure)
    main(0.95, 100, 100, 0.01, (3, 4))
