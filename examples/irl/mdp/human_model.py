from numpy import *

grid_h = 10
h_pos = range(0,grid_h) 
h_action = [0,1,2]

class Human:

    def __init__(self, pos=0, act = random.randint(0,3)):
        self.position = pos
        self.action = h_action[act]
        self.forward = True

    def next_state(self):
        if (self.position == grid_h - 1):
            self.forward = False
        if (self.position == 0):
            self.forward = True

        self.action = random.randint(0, 3)
        if (self.forward):
            step = h_action[self.action]
        else:
            step = h_action[self.action]* -1

        self.position = max( min(self.position + step, grid_h - 1), 0)

        return self.position, step

if __name__ == '__main__':
    hh = Human()
    for i in range(40):
        pos,speed = hh.next_state()
        print("position:",pos,"speed:",speed)