import numpy as np
import random
import matplotlib.pyplot as plt
# Training to find the path from the top left corner to bottom right corner in minimum  steps

r=[[-1,-1,-1,-1,-1,-1,-1,-1,-1], #reward matrice
[-1,-1,-1,-1,-1,-1,-1,100,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-20],
[-1,-1,-1,-1,-1,100,-1,-1,-1]]
#probabilty matrice
p=[[0,0,1,3,3,4,6,6,7],[1,2,2,4,5,5,7,8,8],[0,1,2,0,1,2,3,4,5],[3,4,5,6,7,8,6,7,8]]
def take_action(state,action,e):
    #print state,action
    if r[action][state]==100:
        #if c==4:
            #e=0
        done=True
    else:
        done=False
    return p[action][state],r[action][state],done,e

def display():
    z=np.zeros((1,9))
    z[0][state2]=1
    print "Position"
    print np.reshape(z,(3,3))

q= np.zeros((9,4))
a=0.99#learning rate
d=0.9#discount rate
x=[]
y=[]
e=1#exploration rate
for epoch in range(100):
    state=0
    c=0
    done=False
    while done!=True:
        c=c+1
        if random.random()>e:
            action=np.argmax(q[state])
        else:
            action=random.randint(0,3)
        state2,reward,done,e=take_action(state,action,e)
        #print state,action,state2
        q[state,action]=(1-a)*(reward + d*np.max(q[state2])+q[state,action])
        state=state2
        #display()
        #print q
        
    x.append(epoch)
    y.append(c)     
    a*=a
    e*=0.09
    print "Epoch" ,epoch,"Steps Taken",c
plt.plot(x,y)
plt.show()

state=0
print"Testing"
done=False
#print q
while done!=True:
    action=np.argmax(q[state])
    state2,reward,done,_=take_action(state,action,e)
    #print state,action,state2
    state=state2
    display()
