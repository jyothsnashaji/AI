import numpy as np
import random

class NN():
    def __init__(self,ip,op,h1,h2,h3):
        self.ip = ip
        self.op = op
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3

    def relu(self,z):
         return np.maximum(z,0)

    def theta_init(self,theta1,theta2,theta3,theta4):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4

    def theta_rand(self):
        #Bias also included
        self.theta1 = np.random.randn(self.h1,self.ip+1)
        self.theta2 = np.random.randn(self.h2,self.h1+1)
        self.theta3 = np.random.randn(self.h3,self.h2+1)
        self.theta4 = np.random.randn(self.op,self.h3+1)
     
    def ff(self,X):
        self.m =np.size(X,0)
        if (np.size(X,1)!=self.ip):
            raise ValueError("Input Dimension wrong")
        X = np.concatenate((np.ones((self.m,1)), X), axis=1)
        self.a1 = X
        self.z2 =np.matmul( self.a1,np.transpose(self.theta1))
       
        self.a2 = self.relu(self.z2)
        #print "h1 output",self.a2
        self.a2 = np.concatenate((np.ones((np.size(self.a2,0),1)),self.a2), axis=1)
        self.z3 = np.matmul(self.a2,np.transpose(self.theta2))
        
        self.a3 = self.relu(self.z3)
        #print "h2 output",self.a3
        self.a3 = np.concatenate((np.ones((np.size(self.a3,0),1)), self.a3), axis=1)
        self.z4 = np.matmul(self.a3,np.transpose(self.theta3))
       
        self.a4 = self.relu(self.z4)
        #print "h3 output", self.a4
        self.a4 = np.concatenate((np.ones((np.size(self.a4,0),1)), self.a4), axis=1)
        self.z5 = np.matmul(self.a4,np.transpose(self.theta4))
        self.a5 = self.z5
        self.Y  = self.z5
        #print "output",self.Y

    def cost(self,exp_Y):
        
        return np.sum(np.square(self.Y-exp_Y))

    def derivative(self,z):
        z=(z>0).astype(float)
        return z
    
    def mult(self,a,b):
        z=np.zeros((np.size(a,1),np.size(b,1)))
        
        for i in range(np.size(a,1)):
            z[i]=np.multiply(a[0][i],b[0])
        #print a,b,z
        return np.transpose(z)

    def delta(self,exp_Y):
        self.error=np.subtract(self.Y,exp_Y)
        #print self.error[0][0]
        self.del5 =np.multiply(self.error[0][0],self.a4)
       # self.del4 = np.multiply(self.del5,self.theta4,self.derivative(self.z4))
       # self.del3 = np.multiply(self.del4,self.theta3,self.derivative(self.z3))
       # self.del2 = np.multiply(self.del3,self.theta2,self.derivative(self.z2))
       # error=np.subtract(self.Y,exp_Y)
        #self.del5=np.multiply(np.subtract(self.Y,exp_Y),self.z5,self.derivative(self.a5))
        #print "del5",self.del5
        
        self.del4=self.mult(self.a3,self.derivative(self.z4))
        self.del3=self.mult(self.a2,self.derivative(self.z3))
        self.del2=self.mult(self.a1,self.derivative(self.z2))

        #print "del4",self.del4,self.a3,self.derivative(self.z4)
        #print "del3",self.del3
        #print "del2",self.del2
       

        

    def backprop(self):
        DEL5 = np.zeros((self.m,self.op+1))
        DEL4 = np.zeros((self.m,self.h3+1))
        DEL3 = np.zeros((self.m,self.h2+1))
        DEL2 = np.zeros((self.m,self.h1+1))
    def update(self):
        d=a*self.error[0][0]
        self.theta1= np.subtract(self.theta1,np.multiply(self.del2,d))
        self.theta2= np.subtract(self.theta2,np.multiply(self.del3,d))
        self.theta3= np.subtract(self.theta3,np.multiply(self.del4,d))
        self.theta4= np.subtract(self.theta4,np.multiply(self.del5,a))


import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def hello():
    fig = plt.figure("Actual")
    ax = Axes3D(fig)
    # Make data.
    #x=np.arange(-100,100,5)
    #y=np.arange(-200,200,10)
    x=[]
    y=[]
    z=[]
    x=np.arange(-5,5,0.25)
    y=np.arange(-5,5,0.25)
    
    for  i in range(40):
        qw=[[x[i],y[i]]]
        nn.ff(qw)
        z = np.append(z,nn.Y[0][0])
    ax.plot(x,y,z)
	#plt.savefig('C:/Users/admin/Desktop/3Dphoto/actual.png')
    plt.show()

ep = 0
final = 10000
row = 20
a=0.05
nn=NN(2,1,10,10,10)
nn.theta_rand()
#nn.theta1=[[1,4,7],[2,5,8],[3,6,9]]
#nn.theta2=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
#nn.theta3=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
#nn.theta4=[[1,1,1,1]]
x=[]
y=[]
hello()
#qw=np.arange(-2000,2000,0.45)
for ep in range(final):
    print ep
    qw=np.random.randn(1,nn.ip)
    qw=np.multiply(qw,ep)
    for t in range(row):
        #qw=np.random.randn(1,nn.ip)
        #qw=[[1,2]]
              
        exp_Y = np.sin(np.sqrt(np.sum(np.square(qw))))
        #exp_Y=32943
        #print exp_Y
        nn.ff(qw)
        nn.backprop()
        nn.delta(exp_Y)        
        nn.update()
        qw=np.multiply(qw,-1)
        print "t1", nn.theta1,"t2",nn.theta2,"t3",nn.theta3,"t4",nn.theta4
        x.append(t)
        y.append(nn.cost(exp_Y))
        
        t=t+1
plt.plot(x,y)
plt.show()
nn.ff([[1,1]])
print nn.Y
hello()




