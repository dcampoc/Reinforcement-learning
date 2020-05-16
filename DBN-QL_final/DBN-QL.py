# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:27:52 2020

@author: sheida.nozari and Damian
"""
import scipy.io
from typing import NamedTuple
import numpy as np
import math
from environment import Agent,Target,Environment
import time
import random
from scipy.spatial import distance

DEBUG = 0
DISPLAY_ENVIRONMENT = 0
TRAINNING = 1
SAVING = 0

# QLearning parameters
alpha = 0.3
gamma = 0.6
num_episodes = 100 # 30#10000
max_step_per_episode = 500#20


# class to perform time actions
class Timer(object):
    
    def __init__(self):
        self.t = 0
      
    # get current time
    def tic(self):
        self.t = time.perf_counter()
       
    # return time elapsed since tic() in seconds
    def toc(self):
        return time.perf_counter() - self.t
       
    # return time elapsed since tic() in seconds
    def tocMs(self):
        return (time.perf_counter() - self.t)*1000.0
    
# creating a state structure
# this will store a current state or a state from Q-Table
class State(NamedTuple):
    x: float
    y: float
    vx: float
    vy: float
    
# returns the closest DBN state from the current_state
# returns -1 if no closest state
def FindClosestDBNToCurrentState(current_state, DBNTable, maxDist):
    distanceTable = np.sqrt(((DBNTable[:,0] - current_state.x)*(DBNTable[:,0] - current_state.x) + (DBNTable[:,1] - current_state.y)*(DBNTable[:,1] - current_state.y)))
    closest_indexes = np.argwhere(distanceTable < maxDist)
    
    if len(closest_indexes) == 0:
        return -1
    
    minDist = 999
    closest_index = 0
    for i in range(0, len(closest_indexes)):
        if distanceTable[closest_indexes[i]] < minDist:
            minDist = distanceTable[closest_indexes[i]]
            closest_index = closest_indexes[i]       
    
    return closest_index[0]

# returns the closest QTable state from the current_state
# returns -1 if no closest state
def FindClosestQStateToCurrentState(current_state, StateTable, maxDist):      
    if len(StateTable) == 0:
        return -1
    
    distanceTable = np.sqrt(((StateTable[:,0] - current_state.x)*(StateTable[:,0] - current_state.x) + (StateTable[:,1] - current_state.y)*(StateTable[:,1] - current_state.y)))
    closest_indexes = np.argwhere(distanceTable < maxDist)
    
    if len(closest_indexes) == 0:
        return -1
    
    minDist = 999
    closest_index = 0
    for i in range(0, len(closest_indexes)):
        if distanceTable[closest_indexes[i]] < minDist:
            minDist = distanceTable[closest_indexes[i]]
            closest_index = closest_indexes[i]  
        
    return closest_index[0]
 
# update the QValue of the QTable
def UpdateQValue(QTable, idxPreviousState, idxCurrentState, action_to_do, reward):
    global alpha
    global gamma
        
    
    maxQValue = max(QTable[idxCurrentState,:])

    #QTable[idxPreviousState,action_to_do] = QTable[idxPreviousState,action_to_do]+alpha*(reward + gamma * maxQValue - QTable[idxPreviousState,action_to_do])   
    QTable[idxPreviousState,action_to_do] = (1 - alpha) * QTable[idxPreviousState,action_to_do] + alpha*(reward + gamma* maxQValue)
# used to have same start positions during testing    
def GetTestPositions(num_episodes):
    
    positionsTable = np.empty((0,6), float)
    
    xpositions = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]
    ypositions = [-18, -14, -18, -14, -18, -14, -18, -14, -18, -14, -18]
    
    for i in range(0,len(xpositions)):
        newline = np.zeros((1,6), float) 
        newline[0][0] = xpositions[i]
        newline[0][1] = ypositions[i]
                
        positionsTable = np.vstack((positionsTable, newline))
    
    return positionsTable

# return the best action to do for selected state
def GetBestActionToDoForState(QTable, idxState):
    
    max_entries = np.where(QTable[idxState,:] == max(QTable[idxState,:]))[0]
    number_max_entries = len(max_entries)
    action = max_entries[random.randint(0,number_max_entries-1)]     

    # action = np.argmax(QTable[idxState,:])
            
    return action

# load DBN data into DBNTable that contains x,y,vx,vy of each DBN states
# load COV matrixes of each DBN states into DBNCOVMatrixes
def LoadDBNData(filename):
    
    mat = scipy.io.loadmat(filename)
    # get the name of the saved data (not __global__, __header__, etc...)
    statesIdx = 0
    covIdx = 0
    names = list(mat.keys())
    for i in range(0,len(mat.keys())):
        if names[i] == "Cov_Action":
            covIdx = i
        if names[i] == "U":
            statesIdx = i
            
    DBNTable = np.empty((0,4), float)
    states = mat[list(mat.keys())[statesIdx]]
    # populate the DBNTable
    # for each column
    for i in range(0,len(states)):
        px = states[i][0] # first column
        py = states[i][1] # second column
        vx = states[i][2] # third column
        vy = states[i][3] # fourth column
        DBNTable = np.append(DBNTable, np.array([[px,py,vx,vy]]), axis=0)        
            
    covMatrixes = mat[list(mat.keys())[covIdx]]
    DBNCOVMatrixes = []
    for i in range(0,len(covMatrixes)):
        DBNCOVMatrixes.append(covMatrixes[i])
    
    DBNCOVMatrixes = np.asarray(DBNCOVMatrixes)
    
    return DBNTable, DBNCOVMatrixes

# load transition matrix from DBN data
def loadDBNTransition(filename):
    
    mat = scipy.io.loadmat(filename)
    
    matIdx = 0
    names = list(mat.keys())
    for i in range(0,len(mat.keys())):
        if names[i] == "transitionMat":
            matIdx = i
            break
        
    DBNTransition = mat[list(mat.keys())[matIdx]]    
    
    return DBNTransition

# compute Mahalanobis distance between DBN and agent action
def ComputeMahalanobisDistance(agent_action, dbn_action, cov_matrix):
    inv_matrix = np.linalg.inv(np.asarray(cov_matrix))
    agent_vector = np.asarray(agent_action)
    dbn_vector = np.asarray(dbn_action)
    # normalise the action?
    d = distance.mahalanobis(agent_vector/np.linalg.norm(agent_vector), dbn_vector/np.linalg.norm(dbn_vector), inv_matrix)
    #print(str(d))
    return d

# add state to QTable and StateTable
def AddCurrentStateToQTable(StateTable, QTable, current_state):
    
    QTable = np.append(QTable, np.array([[0,0,0,0,0,0,0,0]]), axis=0) 
    
    StateTable = np.append(StateTable, np.array([[current_state.x, current_state.y]]), axis=0) 
    
    return StateTable, QTable

# get the velocities to apply to the agent based on the action
# 0 up
# 1 down
# 2 right
# 3 left
# 4 up left
# 5 up rigth
# 6 down left
# 7 down right
def GetVelocityValuesFor(action, dbn_velocity):
    vel = dbn_velocity #0.5 # 0.2   
    ang_45_rad = math.radians(45)
    vel_x = vel*math.cos(ang_45_rad)
    vel_y = vel*math.sin(ang_45_rad)
    
    if action == 0: # up is Y+
        return [0,vel]
    elif action == 1: # down is Y-
        return [0,-vel]
    elif action == 2: # right is X+
        return [vel,0]
    elif action == 3: # left is X-
        return [-vel,0]
    elif action == 4: # up left
        return [-vel_x, vel_y]
    elif action == 5: # up right
        return [vel_x,vel_y]
    elif action == 6: # down left
        return [-vel_x,-vel_y]
    elif action == 7: # down right
        return [vel_x,-vel_y]
    
def ComputeRewardPosition(previous_idx, current_idx, DBNTransition):
        
    reward = 0
    
    # we are in the same zone
    if previous_idx == current_idx:
        reward = 0
        
    # not in good zone
    if DBNTransition[previous_idx,current_idx] == 0:
        reward = -10 #-5
    
    previousZoneProb = DBNTransition[previous_idx,previous_idx]
    
    # sort the probabilities from highest to lowest
    sortedProb = -np.sort(-DBNTransition[previous_idx,:])
    
    # take maximum probabilty of second zone after current zone
    if sortedProb[0] == previousZoneProb:        
        maxP = sortedProb[1]
    else:
        maxP = sortedProb[0]
            
    # agent went to best zone
    if DBNTransition[previous_idx,current_idx] == maxP:
        reward = 10 #5
    
    # agent went to not the best zone
    if DBNTransition[previous_idx,current_idx] < maxP and DBNTransition[previous_idx,current_idx] > 0:
        reward = 7 #3
        
    return reward

def GetDBNVelocity(DBNTable, closestIndexDBNTable):
    
    velx = DBNTable[closestIndexDBNTable,2]
    vely = DBNTable[closestIndexDBNTable,3]
    
    vel = math.sqrt(velx*velx+vely*vely)
    
    return vel
    

####################################################################################################
# program starts here
####################################################################################################

env = Environment()
env.CreateEnvironment()

drone = Agent(0,0,n = 'drone')
env.AddAgent(drone)

chosen_dbn = Agent(0,0, c = 'k', n = 'choosen')
env.AddAgent(chosen_dbn)

target = Target(0,15)
env.AddTarget(target)

#car = Agent(0,0,'g','s',10,n = 'car')

reward = 0
# minimum distance we have to consider to be on the target
targetThreshold = 0.8 #2.0#0.5

bad_ending = 0

boundaries = 5

if TRAINNING == 1:
    DBNTable, DBNCOVMatrixes = LoadDBNData("DBNData.mat")
    QTable = np.empty((0,8), float)
    StateTable = np.empty((0,2), float)
    DBNTransition = loadDBNTransition("transitionMat.mat")
    RewardTable = []
    BestAction = []
    
        
if DISPLAY_ENVIRONMENT == 1:
    timeThreshold = 50
else:
    timeThreshold = 0.05
    
startingStateNumber = 0

stepTimer = Timer()
stepTime = 0

totalTime = 0

findClosestTimer = Timer()
findClosestTime = 0

updateQTableTimer = Timer()
updateQTableTime = 0

addStateTimer = Timer()
addStateTime = 0

computeReward2Timer = Timer()
computeReward2Time = 0

stepReward = 0

totalTraveledDistance = 0

episodeTravelDistance = 0

# min maha distance = 0.188641747 with normalised actions
maha_threshold = 5.0

neighborhood_threshold = 1.0

for episode in range(num_episodes):
    if (episode+1)%10 == 0:
        DISPLAY_ENVIRONMENT = 1
        epsilon = 1
    else:
        DISPLAY_ENVIRONMENT = 0    
        epsilon = 0.1*(episode+1)%10
        
    print("Episode " + str(episode+1))    
    
    drone.velocityX = 0
    drone.velocityY = 0
    
    if TRAINNING == 1:
        # choose a random starting position for each episode
        drone.x = random.uniform(-15,15)#(-env.xSize/2+boundaries, env.xSize/2-boundaries)
        drone.y = random.uniform(-19,-13)#-env.ySize/2+boundaries, env.ySize/2-boundaries)
        
        #drone.x = -2.7449718582778875
        #drone.y = -18.93047741585115
        #drone.x = -10
        #drone.y = -15
    
    else:
        drone.x = TestPositionsTables[episode,0]
        drone.y = TestPositionsTables[episode,1]
    
    if DEBUG == 1:
        print("starting at " + str(drone.x) + " " + str(drone.y))

    if TRAINNING == 1:
        dx2 = (target.x - drone.x) * (target.x - drone.x)
        dy2 = (target.y - drone.y) * (target.y - drone.y)
        
        # if we start too close to the target we find an other start position
        while math.sqrt(dx2+dy2) < 3.0:
            drone.x = random.uniform(-15,15)#(-env.xSize/2+boundaries, env.xSize/2-boundaries)
            drone.y = random.uniform(-19,-13)#(-env.ySize/2+boundaries, env.ySize/2-boundaries)
            
            dx2 = (target.x - drone.x) * (target.x - drone.x)
            dy2 = (target.y - drone.y) * (target.y - drone.y)
    
    current_state = State(drone.x,drone.y,drone.velocityX,drone.velocityY)
        
    previous_state = current_state
    
    start_state = current_state
    
    current_step = 0        
    reachTarget = 0
    reward1 = 0        
    findClosestTime = 0
    updateQTableTime = 0
    addStateTime = 0    
    
    episodeTime = 0
    episodeReward = 0    
    episodeTravelDistance = 0
    
    # we do the steps
    stepTimer.tic()
    while current_step < max_step_per_episode:    
        current_step  += 1
        if DISPLAY_ENVIRONMENT == 1:
            env.Update() 
        
        if TRAINNING == 1:
            
            reward_action = 0
 
            findClosestTimer.tic()
            
            closestIndexQTable = FindClosestQStateToCurrentState(current_state,StateTable,neighborhood_threshold)
            
            
            closestIndexDBNTable = FindClosestDBNToCurrentState(current_state,DBNTable,999)
                
            findClosestTime = findClosestTime + findClosestTimer.toc()
            
            # when a QTable state is close enough 
            if closestIndexQTable >= 0:  
                if random.uniform(0,1) > epsilon:
                    action_to_do = random.randint(0,7)   
                else:
                    action_to_do = GetBestActionToDoForState(QTable, closestIndexQTable)  
 
                dbn_velocity = GetDBNVelocity(DBNTable, closestIndexDBNTable)
                agent_action = GetVelocityValuesFor(action_to_do, dbn_velocity)
                #what about if there is no DBN neighborhood?
                if closestIndexDBNTable >= 0:
                    dbn_action = [DBNTable[closestIndexDBNTable,2], DBNTable[closestIndexDBNTable,3]]
                    dbn_matrix = DBNCOVMatrixes[0,closestIndexDBNTable]
                    maha_dist = ComputeMahalanobisDistance(agent_action,dbn_action,dbn_matrix)
                    reward_action = 30 - maha_dist 
                chosen_dbn.SetPosition(DBNTable[closestIndexDBNTable,0],DBNTable[closestIndexDBNTable,1])
            # when a DBN state is close enough
            elif closestIndexDBNTable >= 0: 
                # take random action
                action_to_do = random.randint(0,7)   
                dbn_velocity = GetDBNVelocity(DBNTable, closestIndexDBNTable)  # Magnitude of velocity from the DBN 
                agent_action = GetVelocityValuesFor(action_to_do, dbn_velocity) # Random velocity with the magnitude of the expected DBN
                dbn_action = [DBNTable[closestIndexDBNTable,2], DBNTable[closestIndexDBNTable,3]]
                dbn_matrix = DBNCOVMatrixes[0,closestIndexDBNTable]
                maha_dist = ComputeMahalanobisDistance(agent_action,dbn_action,dbn_matrix)
                
                reward_action = 30 - maha_dist 
                chosen_dbn.SetPosition(DBNTable[closestIndexDBNTable,0],DBNTable[closestIndexDBNTable,1])
                StateTable, QTable = AddCurrentStateToQTable(StateTable,QTable,current_state) # The current state is added to the table
            # we don't have any neighbor
            else:
                # take random action
                action_to_do = random.randint(0,7)   
                # what is the reward_action if we have no neighborhood?
                
        #print("dbn vel: " + str(dbn_velocity)) 
        # perform action
        drone.velocityX = agent_action[0]
        drone.velocityY = agent_action[1]
        
        if drone.velocityY<0 and epsilon==1:
            problem = 1
         
        # get the new position of the agent    
        drone.UpdatePosition()
        previous_state = current_state
        current_state = State(drone.x,drone.y,drone.velocityX,drone.velocityY)      
        
        dx2 = (previous_state.x - current_state.x) * (previous_state.x - current_state.x)
        dy2 = (previous_state.y - current_state.y) * (previous_state.y - current_state.y)
        episodeTravelDistance = episodeTravelDistance + math.sqrt(dx2+dy2)
        
        # update current_state
        #previous_state = current_state
        
        computeReward2Timer.tic()
        prev_dbn_closestIdx = FindClosestDBNToCurrentState(previous_state,DBNTable,999)
        curr_dbn_closestIdx = FindClosestDBNToCurrentState(current_state,DBNTable,999)
        reward_position = ComputeRewardPosition(prev_dbn_closestIdx, curr_dbn_closestIdx, DBNTransition)
        computeReward2Time = computeReward2Timer.toc()
        
        outOfBounds = False
        
        reward_env = 0
        
        # if agent is out of bounds get bad reward
        if current_state.x > env.xSize / 2 or current_state.x < -env.xSize / 2 or current_state.y > env.ySize / 2 or current_state.y < -env.ySize / 2:
            reward_env = -20 
            outOfBounds = True        
        
        reward = reward_action  + reward_env
        #print("rewards: " + str(reward) + " " + str(reward_action) + " " + str(reward_position))
        
        # add the value to the reward table
        reward_updated = True
        if len(RewardTable) < len(QTable):
            RewardTable.append(reward)
            BestAction.append(action_to_do)
        elif RewardTable[closestIndexQTable] < reward:
            RewardTable[closestIndexQTable] = reward
            BestAction[closestIndexQTable] = action_to_do
            
        episodeReward = episodeReward + reward      

        if TRAINNING == 1:            
            updateQTableTimer.tic()
            
            idxPreviousState = FindClosestQStateToCurrentState(previous_state, StateTable, neighborhood_threshold)
            idxCurrentState = FindClosestQStateToCurrentState(current_state, StateTable, neighborhood_threshold)
            
            #print("update: " + str(idxPreviousState) + " " +str(idxCurrentState))
           
            # we update the QTable if the maximum reward for this state has not been reached
            # or if we just added this new state
            if  reward_updated:
                UpdateQValue(QTable, idxPreviousState, idxCurrentState, action_to_do, reward)
                        
            updateQTableTime = updateQTableTime + updateQTableTimer.toc()
        
        # we the agent was out of bounds we reset its position to start position
        if outOfBounds:
            current_state = start_state
            drone.x = start_state.x
            drone.y = start_state.y
            drone.velocityX = start_state.vx
            drone.velocityY = start_state.vy
            continue

        # check if we are on target
        dx2 = (target.x - current_state.x) * (target.x - current_state.x)
        dy2 = (target.y - current_state.y) * (target.y - current_state.y)
        
        if DEBUG == 1:
            print("Target at " + str(math.sqrt(dx2+dy2)))
        
        # target is reached
        # we reset the drone state to the start state
        if max_step_per_episode == current_step:
            print('OBJECTIVE NOT REACHED :(')
        
        if math.sqrt(dx2+dy2) < targetThreshold:
            reachTarget = reachTarget + 1
            # get the time of this step
            totalTime = totalTime + stepTimer.toc() - findClosestTime - updateQTableTime - addStateTime - computeReward2Time
            episodeTime = episodeTime + stepTimer.toc() - findClosestTime - updateQTableTime - addStateTime - computeReward2Time
            # reset the step
            current_state = start_state
            previous_state = start_state
            drone.x = start_state.x
            drone.y = start_state.y
            drone.velocityX = start_state.vx
            drone.velocityY = start_state.vy
            #current_step = current_step + 1
            findClosestTime = 0
            updateQTableTime = 0
            addStateTime = 0   
            reward = 0
            stepTimer.tic()
            print('OBJECTIVE REACHED!!!!!')
            
            break
        '''
        if (stepTimer.toc() - findClosestTime - updateQTableTime - addStateTime - computeReward2Time) > timeThreshold:
            # reset the step
            totalTime = totalTime + stepTimer.toc() - findClosestTime - updateQTableTime - addStateTime - computeReward2Time
            episodeTime = episodeTime + stepTimer.toc() - findClosestTime - updateQTableTime - addStateTime - computeReward2Time
            current_state = start_state
            previous_state = start_state
            drone.x = start_state.x
            drone.y = start_state.y
            drone.velocityX = start_state.vx
            drone.velocityY = start_state.vy
            current_step = current_step + 1
            findClosestTime = 0
            updateQTableTime = 0
            addStateTime = 0   
            reward = 0
            stepTimer.tic()
            '''
        # on this episode we couldn't reach the target in max_step_per_episode steps
        if current_step == max_step_per_episode and reachTarget == 0:
            normaliseReward = episodeReward / episodeTravelDistance
            normaliseTime = episodeTime / episodeTravelDistance
            totalTraveledDistance = totalTraveledDistance + episodeTravelDistance
            print("    Drone did not reached the target")
            print("    Normalized reward for this episode: " + str(normaliseReward))
            print("    Time for this episode: " + str(episodeTime) + " s")
            print("    Drone traveled: " + str(episodeTravelDistance) + " m")
            #print("    Episode speed: " + str(episodeTravelDistance / episodeTime) + " m/s")
            bad_ending = bad_ending + 1
        elif current_step == max_step_per_episode and reachTarget >= 0:
            normaliseReward = episodeReward / episodeTravelDistance
            normaliseTime = episodeTime / episodeTravelDistance
            totalTraveledDistance = totalTraveledDistance + episodeTravelDistance
            # if the agent reached the target during this episode
            print("    Drone reached the target " + str(reachTarget) + " time this episode")
            print("    Normalized reward for this episode: " + str(normaliseReward))
            print("    Time for this episode: " + str(episodeTime) + " s")
            print("    Drone traveled: " + str(episodeTravelDistance) + " m")
            print("    Episode speed: " + str(episodeTravelDistance / episodeTime) + " m/s")
            
print(str(bad_ending) + " bad endings in " + str(num_episodes) + " episodes")
print("Total time for training: " + str(totalTime) + " s")
print("Training speed: " + str(totalTraveledDistance / totalTime) + " m/s")


if SAVING == 1:
    # display the DBN states and the added states
    #env.DisplayStates(StateTable, startingStateNumber)
    
    # we save the data of this training
    #trainingName = str(alpha)+'_'+str(gamma)+'_'+str(num_episodes)+'_'+str(max_step_per_episode)
    
    np.save("QTable_DC.npy", QTable)
    np.save("StateTable_DC.npy", StateTable)
    
    #env.Save("data/env_"+trainingName)

#input("Press Enter to quit...")
























