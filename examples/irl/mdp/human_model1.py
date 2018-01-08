from numpy import *
h_pos = range(0,10) 
h_action = [0,-1,-2]

class Human:

    def __init__(self):
        self.position = 9
        self.action = h_action[random.randint(0,3)]

    def next_state(self):
        self.position = max(self.position + h_action[self.action],0)
        self.action = random.randint(0,3)
        return self.position, self.action
if __name__ == '__main__':
    hh = Human()
    for i in range(20):
        pos,speed = hh.next_state()
        print("position:",pos,"speed:",speed)
