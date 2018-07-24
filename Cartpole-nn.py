import numpy as np 
import tensorflow as tf 
import gym
import random

env=gym.make("CartPole-v0")
gamma = 0.99
e=1
state=env.reset()

input_num_units = 4
hidden_num_units1 = 256
hidden_num_units2 = 256
output_num_units = 2
tf_x = tf.placeholder(tf.float32, [None, input_num_units],name="Input")
tf_exp_q =  tf.placeholder(tf.float32,[None,2],name="Expected_Q_value")
hidden_layer1 = tf.layers.dense(tf_x, hidden_num_units1, tf.nn.relu)
hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_num_units2, tf.nn.relu)
output_layer = tf.layers.dense(hidden_layer2, output_num_units)
cost = tf.losses.mean_squared_error(tf_exp_q, output_layer)
train_op = tf.train.GradientDescentOptimizer(0.005).minimize(cost)


replay_memory=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    state=np.reshape(state,[1,4])

    for i in range(100000):
        print i
        action= sess.run(output_layer,{tf_x:state}) 
        action = np.argmax(action[0])
        
        if random.random()<e:
            action = random.randint(0,1)
        prev_state = state
        state, reward, done, _ = env.step(action)
        state = np.reshape(state, [1, 4])
        if done!=True:
            
            replay_memory.append([prev_state, action, reward, state])
            e=e*0.9
        else:
            reward=-20
            replay_memory.append([prev_state, action, reward, state])
            state=env.reset()
            state=np.reshape(state,[1,4])
        

    for i in range(100000):
        minibatch=np.random.randint(0,100000,50)

        print i
        for record in minibatch:
            
            rec=replay_memory[record]
            prev_state=rec[0]
            action=rec[1]
            reward=rec[2]
            state=rec[3]
        
            expected=reward+gamma*np.max(sess.run(output_layer,{tf_x:state})[0])
            target=(sess.run(output_layer,{tf_x:prev_state}))
            target[0][action]=expected
            sess.run(train_op,{tf_x:state,tf_exp_q:target})

    
   
    for episode in range(10):
        c=0
        done=False
        state=env.reset()
        while done!=True:
            state=np.reshape(state,[1,4])
            action= sess.run(output_layer,{tf_x:state}) 
            action = np.argmax(action[0])
            state, reward, done, _ = env.step(action)
            c=c+1
        print episode,c
