
import gym
import math

import matplotlib.pyplot as plt
import numpy as np
import random

def getindex(state):
 index=[]
 
 for i in range((len(state))):
  if state[i]<=bounds[i][0]:
   k=0
   reward= -20
  elif state[i]>=bounds[i][1]:
   k=(discrete[i] -1)
   reward=-20
  else:
   w=bounds[i][1]-bounds[i][0]
   per_index_width = w/(discrete[i]-1)
   k = int (round ((state[i]-bounds[i][0])/per_index_width))
  index.append(k)
 #print index
 return tuple(index)


env = gym.make("CartPole-v0")


exp=1
bounds = list(zip(env.observation_space.low,env.observation_space.high))
bounds[0]=(-2.2,2.2)
bounds[1]=(-0.5,0.5)
bounds[2]=(-math.radians(11),math.radians(11))#check dtype
bounds[3]=(-math.radians(50),math.radians(50))
discrete=(7,7,7,7)

q=np.zeros(discrete+(2,))

t=0
x=[]
y=[]
discount= 0.99

learning_rate = 0.834
for epoch in range(100):
 count=0
 for episode in range(100):
  
  done=False
  state=env.reset()
  state_index=getindex(state)
  while done!=True:
   if random.random()>exp:
    action=np.argmax(q[state_index])
   else:
    action = env.action_space.sample()
   state2,reward,done,_= env.step(action)
   if done==True:
    reward=-75
   state2_index= getindex(state2)
   q[ state_index+(action,)]+=learning_rate* (reward + discount*np.max(q[state2_index])-q[ state_index+(action,)])
   state_index= state2_index
   count+=1
  exp*=0.8
  learning_rate=pow(learning_rate,0.85)
 #plt.plot(epoch,count/100)
 print "Epoch no:", epoch,"Average Reward over the 100 episodes", count/100
 x.append(epoch)
 y.append(count/100)
 t+=count
 
#print "Avg reward per episode",t/10000
#plt.plot(x,y)
#plt.show()


for i in range(1):
 done=False
 count = 0
 state=env.reset()
 state_index=getindex(state)
 
 while done!=True:
 
  action=np.argmax(q[state_index])
 
  state2,reward,done,info= env.step(action)
  if done==True:
   reward=-160
  state2_index= getindex(state2)
  q[ state_index+(action,)]+=learning_rate* (reward + discount*np.max(q[state2_index])-q[ state_index+(action,)])
  state_index= state2_index
  count+=1
 print count


