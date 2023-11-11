from FourRooms import FourRooms
import numpy as np
fourRoomsObj = FourRooms('simple')
epsilon = 0.8
ds = 0.8 #discount factor for future rewards
learning_rate = 0.6
environment_rows=13
environment_columns=13
q_values = np.zeros((environment_rows, environment_columns, 4))

#method to get the next action
def action(current_row_index, current_column_index):
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: #choose a random action
    return np.random.randint(4)



#main method that pefoms all the training and prints final run
def main():
    global epsilon
    yn=input("-stochastic? [Y/N] \n")
    if((yn=='Y')or(yn=='y')):
        epsilon=0.8

    for episode in range(1000):
        row_i=FourRooms.getPosition(fourRoomsObj)[0]
        column_i=FourRooms.getPosition(fourRoomsObj)[1]
        while((FourRooms.isTerminal(fourRoomsObj)==False)):
            action_index=action(row_i,column_i)
            old_row_index, old_column_index = row_i, column_i
            cellType, newPos, packagesLeft, isTerminal = fourRoomsObj.takeAction(action_index)
            row_i=newPos[0]
            column_i=newPos[1]
            reward=-1
            if(FourRooms.isTerminal(fourRoomsObj)):
                reward=-100
            if((cellType>0)or(packagesLeft==0)):
                reward=100
            old_q = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (ds * np.max(q_values[row_i, column_i])) - old_q
            new_q = old_q + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q
        FourRooms.newEpoch(fourRoomsObj)
    FourRooms.showPath(fourRoomsObj,999,"scenario1.png")
    


if __name__ == "__main__":
    main()
