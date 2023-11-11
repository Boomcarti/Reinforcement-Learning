from FourRooms import FourRooms
import numpy as np
fourRoomsObj = FourRooms('multi')
environment_rows=13
environment_columns=13
lr = 0.6
q_vals= [np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),
                np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4))]

discount_factor = 0.7
epsilon = 0.85

booleanflags = [False, False, False]
#muktple q tables for the different combinations of package collection

epi=0
def boolindex():
    f=4*booleanflags[0] 
    lv=booleanflags[2] 
    return f+ 2*booleanflags[1] + lv

#method to assign reward
def rewardfunction(cellType):
    if (cellType == 0):
        return -1
    elif (cellType > 0):
        if (not booleanflags[cellType-1]):
            booleanflags[cellType-1] = True
            return 100
        else:
            return -1
    elif(cellType < 0):
        return -100
        


def flipflags():
    global booleanflags
    booleanflags[2] = False
    booleanflags[0] = False
    booleanflags[1] = False



def main():
    global epsilon
    yn=input("-stochastic? [Y/N] \n")
    if((yn=='Y')or(yn=='y')):
        epsilon=0.8
    for episode in range(1000):
        i = 0
        epi =episode

        FourRooms.newEpoch(fourRoomsObj)
        column_i=FourRooms.getPosition(fourRoomsObj)[1]
        row_i=FourRooms.getPosition(fourRoomsObj)[0]
        
        while((FourRooms.isTerminal(fourRoomsObj)==False)):
   
    
            action_i=0
            current_r, current_c=row_i,column_i
            buffer=(epi/1000)*(1-epsilon)
            if np.random.random() < epsilon+buffer:
              action_i= np.argmax((q_vals[i])[current_r, current_c])
            else: #choose a random action
              action_i= np.random.randint(4)
            old_row_i, old_column_i = row_i, column_i
            
            cellType, newPos, packagesLeft, isTerminal = fourRoomsObj.takeAction(action_i)
            reward = rewardfunction(cellType)
            row_i=newPos[0]
            column_i=newPos[1]
            old_q_value = (q_vals[i])[old_row_i, old_column_i, action_i]
            neq = np.max((q_vals[i])[row_i, column_i])
            new_q_value = (1-lr)*old_q_value+lr*(reward+discount_factor*neq)
            (q_vals[i])[old_row_i, old_column_i, action_i] = new_q_value
            if (reward > 0):
                i = boolindex()
            if (packagesLeft == 0):
                break
            if (isTerminal):
                break
        flipflags()




    
    FourRooms.showPath(fourRoomsObj,-1,"scenario2.png")
    


if __name__ == "__main__":
    main()