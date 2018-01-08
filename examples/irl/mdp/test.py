import numpy as np
import matplotlib.pyplot as plt


# def int_to_point(i):
#   """
#   Convert a state int into the corresponding coordinate.


def main():
    list1 = [10, 50, 100, 200, 300, 400, 500]
    list2 = [11, 51, 102, 205, 366, 40, 50]
    lin1, = plt.plot(list1, list1, "b-", label='Line 2')
    lin2, = plt.plot(list1, list2, "r--", label='Line 1')
    plt.title("Error in policy")
    plt.xlabel("Sample sizes")
    plt.ylabel("Error (%)")
    plt.legend([lin1, lin2], ['Line Up', 'Line Down'])
    plt.show()


# data = np.loadtxt("pass_disc_break_variables.txt",skiprows=0)
# data = data.astype(np.int64)
#
# temp1 = data[:,0]
# temp2 = data[:,1]
# temp3 = data[:,2]+5
# temp4 = data[:,3]
# temp5 = data[:,4]

# car_velo = data[:,5]

# print(data[0:10,0])


main()
