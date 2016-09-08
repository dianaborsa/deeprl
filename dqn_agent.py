# ------------------------------
# DQN implementation
# ------------------------------

import tensorflow as tf
import memory 
import random
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


class DQN():

	# ------------------------------
	# class constructor 
	# ------------------------------
	def __init__ (self, in_size, out_size, replayMemorySize, discount =0.99, learningRate = 0.01):
		
		self.in_size      = in_size
		self.nActions     = out_size
		self.replayMemory = memory.Memory()
		self.discount     = discount
		self.learningRate = learningRate

		# get network
		self.initializeNetwork()

		self.var_online_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'online_net_scope')
		self.var_target_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_net_scope')

		if len(self.var_online_net) != len(self.var_target_net):
			print('Target and Online networks do not match in size!!!')
			print('Size of onlineNetwork: %d' %len(self.var_online_net))
			print('Size of targetNetwork: %d' %len(self.var_target_net))
		else:
			print('Agent succesfully constructed')

		self.getTarget_Op()

	def initializeNetwork(self):

		# placeholders
		self._input  = tf.placeholder(tf.float32, shape=[None, self.in_size])
		self.a       = tf.placeholder(tf.float32, shape=[None, self.nActions] ) 
		self.y_      = tf.placeholder(tf.float32, shape=[None])


		with tf.variable_scope('online_net_scope'):
			print('Initializing onlineNetwork')
			model = self.getModel(self.in_size, self.nActions)
			self.onlineNetwork = model
			#self.onlineQ_sa = tf.reduce_sum(tf.mul(self.onlineNetwork,self.a) , reduction_indices=[1])


		with tf.variable_scope('target_net_scope'):
			print('Initializing targetNetwork')
			targetNetwork = self.getModel(self.in_size, self.nActions)
			self.targetNetwork = targetNetwork
			#self.targetV = tf.reduce_max(self.targetNetwork, reduction_indices=[1]) 

	def getOneHotForActions(self, actions):
		actions_onehot = np.zeros((len(actions), self.nActions))
		actions_onehot[xrange(len(actions)), np.array(actions,int)] = 1
		return actions_onehot

	def getPhiTerminal(self, is_done):
		return np.array(is_done, int)

	def getTarget_Op(self):
		
		V_target    = tf.reduce_max(self.targetNetwork, reduction_indices=[1]) 
		Q_sa_online = tf.reduce_sum(tf.mul(self.onlineNetwork,self.a) , reduction_indices=[1])
	
		# compute targets operation
		self._rewards   = tf.placeholder(tf.float32, shape=[None])
		self._terminals = tf.placeholder(tf.float32, shape=[None]) 
		
		self.target_op = self._rewards+self.discount*V_target*(1-self._terminals)

		self.loss = tf.reduce_sum(tf.square(self.y_ - Q_sa_online))

	def getMiniBatch(self, batchSize):
		(cstates, actions, rewards, nstates, done) = self.replayMemory.getMiniBatch(batchSize)
		phi_s    = cstates
		phi_a    = self.getOneHotForActions(actions)
		rewads   = rewards
		phi_ns   = nstates
		terminal = self.getPhiTerminal(done)
		# targets  = sess.run(self.target_op, feed_dict={x: phi_ns, r: rewards, _terminal: terminal})
		# return (phi_s, phi_a, targets)

		return (phi_s, phi_a, rewards, phi_ns, terminal)

	def getModel(self, dim_input, dim_ouput):

		# first layer
		W1 = weight_variable([dim_input, 300])
		b1 = bias_variable([300])

		first_layer = tf.nn.relu(tf.matmul(self._input,W1)+b1)

		# second layer
		W2 = weight_variable([300,100])
		b2 = bias_variable([100])

		second_activation = tf.matmul(first_layer, W2) + b2
		second_layer = tf.nn.relu(second_activation)

		# third layer
		W21 = weight_variable([100,50])
		b21 = bias_variable([50])

		second1_activation = tf.matmul(second_layer, W21) + b21
		second1_layer = tf.nn.relu(second1_activation)

		#output layer
		W3 = weight_variable([50,dim_ouput])
		b3 = bias_variable([dim_ouput])

		third_activation = tf.matmul(second1_layer, W3) + b3
		#third_layer = tf.nn.relu(third_activation)
		third_layer = third_activation
		# y = tf.reduce_sum(tf.mul(third_layer,self.a) , reduction_indices=[1])

		# model = {"state_placeholder" : x,
		# 		 "Q_s" : third_layer,
		# 		 "Q_sa": y,
		# 		 "target_placeholder": y_,
		# 		 "action_placeholder": a}

		return third_layer


	def updateTargetNetwork(self, sess):
		for variableIndex in xrange(len(self.var_online_net)):
			sess.run(self.var_target_net[variableIndex].assign(self.var_online_net[variableIndex]))


	def egreedy_policy(self, net, state, epsilon=0.0):
		Q_values = net.eval(feed_dict = {self._input:[state]})[0]

		if random.random() < epsilon:
			return random.randint(0,self.nActions - 1)
		else:
			return np.argmax(Q_values)

	def behaivour_policy(self, state, epsilon):
		return self.egreedy_policy(self.onlineNetwork, state, epsilon)




	#def performTraining(self):

def main():
	in_size, out_size, replayMemorySize = 8, 4, 100
	agent = DQN(in_size, out_size, replayMemorySize)
	print agent.onlineNetwork
	print agent.targetNetwork

if __name__ == '__main__':
	main()