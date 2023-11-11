from FourRooms import FourRooms
import numpy as np
fourRoomsObj = FourRooms('rgb')
environment_rows=13
environment_columns=13
learning_rate = 0.6
discount_factor = 0.7
q_vals= [np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),
                np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4)),np.zeros((environment_rows, environment_columns, 4))]

epsilon = 0.85

#muktple q tables for the different combinations of package collection

booleanflags = [False, False, False]
def flipflags():
    global booleanflags
    booleanflags[2] = False
    booleanflags[0] = False
    booleanflags[1] = False

epi=0



#finds next action with use of epsilon decay
def action(current_r, current_c, epsilon,i):
    buffer=(epi/4000)*(1-epsilon)

    if np.random.random() < epsilon+buffer:
      return np.argmax((q_vals[i])[current_r, current_c])
    else: #choose a random action
      return np.random.randint(4)
#main method to train and test RL learner

#to create an index in the qtable list to use the correct table
def boolindex():
    first=4*booleanflags[0]
    second=2*booleanflags[1]
    third=booleanflags[2]
    ans =first+second+third
    return ans
        
#method to assign reward
    
c=1
def main():
    global epsilon
    yn=input("-stochastic? [Y/N] \n")
    if((yn=='Y')or(yn=='y')):
        epsilon=0.8
    global c

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
            global c
            r=0
            if(cellType < 0):
                r=-100
            elif (cellType > 0):
                if (booleanflags[cellType-1]==False):
                    booleanflags[cellType-1] = True
                    if(cellType ==c):
                        r= 100
                        c=c+1
                    else:
                        r= -100
                else:
                    r= -1
            elif (cellType == 0):
                r= -1
            movereward = r
            row_i=newPos[0]
            column_i=newPos[1]
   
            old_q_value = (q_vals[i])[old_row_i, old_column_i, action_i]
  
            neq = np.max((q_vals[i])[row_i, column_i])
            new_q = (1-learning_rate)*old_q_value+learning_rate*(movereward+discount_factor*neq)
            (q_vals[i])[old_row_i, old_column_i, action_i] = new_q
            if (movereward > 0):
                i = boolindex()
            if (packagesLeft == 0):
                break
            if (isTerminal):
                break
        flipflags()
        c=1
    
    FourRooms.showPath(fourRoomsObj,-1,"scenario3.png")
    


if __name__ == "__main__":
    main()