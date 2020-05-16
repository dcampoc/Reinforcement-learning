import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import math
import random

class Agent:
        
    maxXVelocity = 1
    minXVelocity = -1
    
    maxYVelocity = 1
    minYVelocity = -1
    
    
    def __init__(self, posx = 0, posy = 0, c = 'r', m = 'o', s = 5, n = 'agent'):
        self.x = posx
        self.y = posy
        self.color = c
        self.marker = m
        self.size = s
        self.name = n
        self.velocityX = 0 #m/s
        self.velocityY = 0 #m/s
    
    # set x velocity to the agent
    def SetPosition(self,x, y):
        self.x = x
        self.y = y
                
    # set x velocity to the agent
    def SetXVelocity(self,velx):
        self.velocityX = velx
    
    # set y velocity to the agent
    def SetYVelocity(self,vely):
        self.velocityY = vely
                
    def SetXVelocityByAcc(self, accx):
        #self.velocityX = self.velocityX + accx
        self.velocityX = accx
        
        if self.velocityX > self.maxXVelocity:
            self.velocityX = self.maxXVelocity
            
        if self.velocityX < self.minXVelocity:
            self.velocityX = self.minXVelocity
        
    def SetYVelocityByAcc(self, accy):
        #self.velocityY = self.velocityY + accy
        self.velocityY = accy
        
        if self.velocityY > self.maxYVelocity:
            self.velocityY = self.maxYVelocity
            
        if self.velocityY < self.minYVelocity:
            self.velocityY = self.minYVelocity
        
    # update the agent's position with the current agent velocity
    def UpdatePosition(self):
        self.x = self.x + self.velocityX
        self.y = self.y + self.velocityY
        
    def PrintInfos(self):
        print(self.name + " is at position (" + str(self.x) + "," + str(self.y) + ") with velocity (" + str(self.velocityX) + "," + str(self.velocityY) + ")")
    
class Target:
    
    def __init__(self, posx = 0, posy = 0, c = 'b', m = '^', s = 5):
        self.x = posx
        self.y = posy
        self.color = c
        self.size = s    
        self.marker = m

class Environment:

    xSize = 40
    ySize = 40    
    stepTime = 0.001 # 1ms 
    
    agentsList = [] 
    
    def __init__(self, xRange = 40, yRange = 40, t = 0.001):
        plt.ion()
        self.xSize = xRange
        self.ySize = yRange
        stepTime = t
        self.agentsList.clear()
        self.hl, = plt.plot([], [])
        self.fig = plt.figure()
    
    # create an environment at size xSize*ySize
    def CreateEnvironment(self):
        plt.close()
        
        axes = plt.gca()
        axes.set_xlim([-self.xSize/2, self.xSize/2])
        axes.set_ylim([-self.ySize/2, self.ySize/2])
        
    # register an agent to the environment to be able to plot the agent
    def AddAgent(self, agent):
        self.agentsList.append(agent)
            
    # remove an agent from the environment    
    def RemoveAgent(self, agent):
        self.agentsList.remove(agent)
        
    # register an target to the environment
    def AddTarget(self, t):
        self.target = t
    
    # draw an agent on the environment    
    def DrawAgent(self, agent):
        plt.plot(agent.x, agent.y, color=agent.color, marker=agent.marker,
         markerfacecolor=agent.color, markersize=agent.size)
    
    # draw the target on the environment
    def DrawTarget(self, target):
        plt.plot(target.x, target.y, color=target.color, marker=target.marker,
         markerfacecolor=target.color, markersize=target.size)
    
    # update the plotted positions of the agents on the environment
    def Update(self):
    
        # clear the plot
        plt.cla()
        
        # set the environment limits
        axes = plt.gca()
        axes.set_xlim([-self.xSize/2, self.xSize/2])
        axes.set_ylim([-self.ySize/2, self.ySize/2])
        
        # draw all agents
        for agent in self.agentsList:
            self.DrawAgent(agent)
                    
        # draw the target
        self.DrawTarget(self.target)
        
        # update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # sleep
        plt.pause(self.stepTime) #used for spyder integration
        #sleep(self.stepTime)
        
    def DisplayStates(self, StatesTable, startingStateNumber):
    
        # clear the plot
        plt.cla()
        
        # set the environment limits
        axes = plt.gca()
        axes.set_xlim([-self.xSize/2, self.xSize/2])
        axes.set_ylim([-self.ySize/2, self.ySize/2])
        
        for i in range(0, len(StatesTable)):
            stateColor = 'k'
            if i > startingStateNumber-1:
                stateColor = 'g'            
            
            plt.plot(StatesTable[i][0], StatesTable[i][1], color=stateColor, marker='o',
             markerfacecolor=stateColor, markersize=5)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def Save(self, name):        
        plt.savefig(name+".png")



######################################
### test part


# env = Environment()
# env.CreateEnvironment()

# drone = Agent(5,0,n = 'drone')
# car = Agent(0,0,'g','s',10,n = 'car')
# target = Target(10,10)

# env.AddAgent(drone)
# env.AddAgent(car)

# maxStep = 300
# currentStep = 0
# alpha = 0.1

# while currentStep < maxStep:

    # env.Update()
    
    # car.SetXVelocity(alpha*random.uniform(-1.5, 1.5) + (1-alpha)*car.velocityX)
    # car.SetYVelocity(alpha*random.uniform(-1.5, 1.5) + (1-alpha)*car.velocityY)
    
    # if car.x > env.xSize/2 or car.x < -env.xSize/2:
        # car.SetXVelocity(-car.velocityX)
    # if car.y > env.ySize/2 or car.y < -env.ySize/2:
        # car.SetYVelocity(-car.velocityY)
    
    # x2 = (car.x - drone.x)*(car.x - drone.x)
    # y2 = (car.y - drone.y)*(car.y - drone.y)
    # dist = math.sqrt(x2+y2)
    
    # if dist < 0.1:
        # drone.SetXVelocity(-drone.velocityX)
        # drone.SetYVelocity(-drone.velocityY)
    # else:
        # drone.SetXVelocity((car.x - drone.x)/50)
        # drone.SetYVelocity((car.y - drone.y)/50)
        
    # drone.UpdatePosition()
    # car.UpdatePosition()
        
    # currentStep = currentStep + 1

# input("Press Enter to continue...")
