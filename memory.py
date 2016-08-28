# ==================
# Memory class
# ==================

import numpy as np

def sliceExperience(experience, _dim):
	nSamples = len(experience)
	return list(experience[i][_dim] for i in xrange(nSamples))
	
class Memory():

	def __init__(self, experienceBatch=[]):

		# list of experience
		self.MemoryBuffer  = experienceBatch
		self.memSize = len(experienceBatch)


	def updateMemoryBuffer(self, newMemoryBuffer):
		self.MemoryBuffer = newMemoryBuffer
		self.memSize      = len(newMemoryBuffer)

	
	def addExperienceToMemory(self, experience):
		newMemory = self.MemoryBuffer+experience
		self.updateMemoryBuffer(newMemory)


	def getRandomSample(self, subsampleSize=None, new_ordering=None):
		if new_ordering == None:
			new_ordering = np.random.permutation(self.memSize)

		if subsampleSize != None:
			new_ordering = new_ordering[xrange(subsampleSize)]

		return list(self.MemoryBuffer[i] for i in new_ordering)


	def downsizeMemory(self, new_memSize, new_ordering=None):
		if new_memSize > self.memSize:
			print('Cannot downsample current memory as size is smaller than the required subsample size')
			return 0
		else:
			self.updateMemoryBuffer(self.getRandomSample(new_memSize, new_ordering))

	def resetMemory(self):
		self.updateMemoryBuffer([])

	def getMiniBatch(self, batchSize=32):

		miniBatch = self.getRandomSample(batchSize)
		cstates = sliceExperience(miniBatch, 0)
		actions = sliceExperience(miniBatch, 1)
		rewards = sliceExperience(miniBatch, 2)
		nstates = sliceExperience(miniBatch, 3)
		done    = sliceExperience(miniBatch, 4)

		return (cstates, actions, rewards, nstates, done)






