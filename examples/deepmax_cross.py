"""
Run maximum entropy inverse reinforcement learning on chicken crossing game.

Yiqian Gan
ganyq@umich.edu
"""

import numpy as np
import matplotlib.pyplot as plt

import irl.deep_maxent as deep_maxent
from   irl.value_iteration import find_policy
import irl.mdp.crossworld as crossworld    ##### change 

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
    l1 = l2 = 0.00
    grid_size = 10
    grid_size_h = 10
    grid_size_v = 3

    xw = crossworld.Crossworld(discount,wind,True)
    tp = xw.transition_probability
    temp = xw.transition_probability.sum(axis=2)
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
                                            lambda s: policy[s],True)
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

    v = np.zeros(xw.n_states)

    
    # temp = np.dot(tp, reward + discount * v)
    a = 1
    r = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
        xw.n_actions, discount, xw.transition_probability, trajectories, epochs,
        learning_rate, l1=l1, l2=l2)

    recover_policy = find_policy(xw.n_states, xw.n_actions, xw.transition_probability,
                         r, xw.discount, stochastic=False)
    plt.subplot(2, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size_h,grid_size_v))[:,:,0])
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(2, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size_h,grid_size_v))[:,:,0])
    plt.colorbar()
    plt.title("Recovered reward")

    plt.subplot(2, 2, 3)
    plt.pcolor(policy.reshape((grid_size, grid_size_h,grid_size_v))[:,:,0])
    plt.colorbar()
    plt.title("Optimal Policy")
    plt.subplot(2, 2, 4)
    plt.pcolor(recover_policy.reshape((grid_size, grid_size_h,grid_size_v))[:,:,0])
    plt.colorbar()
    plt.title("Recovered Policy")
    plt.show()

if __name__ == '__main__':
    main(0.95, 5000, 5000, 0.01, (3, 3))
