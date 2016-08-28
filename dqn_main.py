import gym
import tensorflow as tf 
import numpy as np 
import random
import dqn_agent as dqn

flags = tf.app.flags


FLAGS = flags.FLAGS
flags.DEFINE_integer('nEpisodes',   5, 'Nr of episodes sampled')
flags.DEFINE_integer('nSamples' ,1000, 'Nr of samples per episode')
flags.DEFINE_integer('nActions' ,   4, 'Nr of actions') 
flags.DEFINE_integer('MAX_ITER' , 500, 'Value/Policy iterations')
flags.DEFINE_integer('nEpochs'  ,  10, 'Nr of epochs per optimization(PI/VI)')
flags.DEFINE_integer('BATCH_SIZE', 32, 'Mini-Batch size')
flags.DEFINE_boolean('training', False, 'Training Flag')
flags.DEFINE_boolean('evaluate', False, 'Evaluation Flag')
flags.DEFINE_integer('random_seed',123, 'Random seed')
flags.DEFINE_integer('learningRateIndex', 1, 'learningRate = 1e(learningRateIndex-6)')


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'MountainCar-v0'
#ENV_NAME = 'CartPole-v1'
#ENV_NAME = 'CarRacing-v0'
#ENV_NAME = 'FlappyBird-v0'
#ENV_NAME = 'Breakout-ram-v0'
#ENV_NAME = 'Riverraid-ram-v0'
#ENV_NAME = 'Skiing-ram-v0'
#ENV_NAME = 'CrazyClimber-ram-v0'
REPLAY_SIZE = 10000 # max replay buffer

ERROR_THRESHOLD= 10^-5

def playoutEpisode(env, agent, nSamples, RENDERING_FLAG=False, epsilon=0.9):
	#enter enviroment
	cstate = env.reset()
	episode = []
	total_reward = 0
	for step in xrange(nSamples):
		action = agent.behaivour_policy(cstate, epsilon)

		if RENDERING_FLAG:
			env.render()
		# take action and observe outcome
		nstate, reward, done, _ = env.step(action)

		total_reward = total_reward+reward

		episode.append((cstate, action, reward, nstate, done))
		cstate = nstate
		if done:
			# terminal reward handling?
			break
	return episode, total_reward

def collectEpisode(env, agent, nSamples, RENDERING_FLAG=False, epsilon=0.9):
	episode, _ = playoutEpisode(env, agent, nSamples, False)
	agent.replayMemory.addExperienceToMemory(episode)

def collectExperience(env, agent, nEpisodes, nSamples):
	for iterEpisode in xrange(nEpisodes):
		collectEpisode(env, agent, nSamples)
		#print('replayMemory Size: ', agent.replayMemory.memSize)
		if agent.replayMemory.memSize>REPLAY_SIZE:
			agent.replayMemory.downsizeMemory(REPLAY_SIZE)

def evaluateNetwork(env, agent, nEpisodes,nSamples):
	# enter enviroment
	cstate = env.reset()
	total_reward = 0
	for i in xrange(nEpisodes):
		_, episode_reward = playoutEpisode(env, agent, nSamples, RENDERING_FLAG = True, epsilon=0.0)
		total_reward += episode_reward

	average_reward = total_reward/nEpisodes
	return average_reward


def main():

	# =================================
	# 0. Sample dataset
	# =================================
	nEpisodes  = FLAGS.nEpisodes
	nSamples   = FLAGS.nSamples 
	maxIter    = FLAGS.MAX_ITER
	nEpochs    = FLAGS.nEpochs
	BATCH_SIZE = FLAGS.BATCH_SIZE
	epsilon    = 0.1

	# initialize OpenAI Gym env and dqn agent
	env = gym.make(ENV_NAME)

	in_dim  = env.observation_space.shape[0]
	out_dim = env.action_space.n 

	agent = dqn.DQN(in_dim, out_dim, REPLAY_SIZE)

	# ==================================
	# 1. Build model
	# ===================================

	model = agent.onlineNetwork
	(x, a, y_) = (model["state_placeholder"], model["action_placeholder"], model["target_placeholder"])
	q_s, y = (model["Q_s"], model["Q_sa"])

	# define loss as the Bellman residual
	loss = tf.reduce_mean(((y_-y)**2))
	normalize_loss = tf.reduce_mean((1-y/y_)**2)
	V = tf.reduce_max(q_s, reduction_indices=[1]) 
	
	# # compute targets operation
	# r = tf.placeholder(tf.float32, shape=[None])
	# discount = 0.99
	# target   = r+discount*V*(1-terminal)

	
	# =====================================
	# 2. Train model
	# =====================================
	train_op = tf.train.AdamOptimizer(learning_rate=(10**(FLAGS.learningRateIndex-5))).minimize(loss)
	#train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
	#train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
	#train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

	nSamplesOpt = 5000
	with tf.Session() as sess: 
		sess.run(tf.initialize_all_variables())

		# first collect some experience
		collectExperience(env, agent, nEpisodes, nSamples)

		# repeat untill convergence
		for valueIter in range(maxIter):

		# 	repeat for k steps 
			for epoch in range(nEpochs):
		# 		sample experience from memoryReplay
				sizeOfWorkingMemory = min(nSamplesOpt, agent.replayMemory.memSize)
				(cstates, actions, rewards, nstates, terminals) = agent.getMiniBatch(sizeOfWorkingMemory)

				with tf.variable_scope('target_net_scope'):
					targets  = sess.run(agent.target_op, feed_dict={agent._input: nstates, agent._rewards: rewards, agent._terminals: terminals})
					#print targets

				for startBatch, endBatch in  zip( range(0,sizeOfWorkingMemory,BATCH_SIZE), range(BATCH_SIZE,sizeOfWorkingMemory,BATCH_SIZE)):
		# 			train/update onlineNetwork
					with tf.variable_scope('online_net_scope'):
						sess.run(train_op, feed_dict={x:cstates[startBatch:endBatch], a:actions[startBatch:endBatch], y_:targets[startBatch:endBatch]})
					
				training_error = sess.run(loss, feed_dict={x:cstates, a:actions, y_:targets})
				if training_error < ERROR_THRESHOLD:
					print('Breaking?')
					break
				# print("Training error: %.6f"%training_error)

		# 		collect experience (online exploratory strategy)
				collectExperience(env, agent, nEpisodes, nSamples)

		# 		add experience to the replayMemory

		# 	update targetNetwork
			print('Updating target network')
			agent.updateTargetNetwork(sess)

		# end

		# 	evaluate learning
			if valueIter%10==0:
				average_reward = evaluateNetwork(env, agent, 3, 500)
				print('Iter %d Current performance %f'%(valueIter,average_reward))
				print("Training error: %f"%training_error)





if __name__ == '__main__':
	
	# # (REPRODUCIBILITY) set random seeds
	# tf.set_random_seed(123)

	tf.flags.FLAGS._parse_flags()	
	tf.set_random_seed(FLAGS.random_seed)
	# #tf.app.run()
	
	main()