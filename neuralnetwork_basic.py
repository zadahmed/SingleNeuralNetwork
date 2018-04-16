from numpy import exp , array , random , dot

class NeuralNetwork():

	def __init__(self):

		#seed the random number so that we can use the same values during each iteration
		random.seed(1)

		# we model a single neuron neural network  , with 3 input connections and one output connection

		self.synaptic_weights = 2 * random.random((3,1)) - 1


		#the sigmoid function , which describes an s shaped curve
		#we pass the sum of weighted inputs through this function to normalize them between 0 and 1
	def __sigmoid(self , x):
		return 1 / ( 1 + exp(-x))


	def think(self , inputs):
		#pass inputs through our neural network to predict the values of the output
		return self.__sigmoid(dot(inputs , self.synaptic_weights))


	def __sigmoid_derivative(self,x):
		return x * ( 1 - x)

	def train(self , trainingset_inputs , trainingset_outputs , numiterations):
		for iteration in xrange(numiterations):
			output = self.think(trainingset_inputs)

			#calculate error

			error = trainingset_outputs - output

			adjustment = dot(trainingset_inputs.T , error * self.__sigmoid_derivative(output))

			self.synaptic_weights += adjustment










if __name__ == '__main__':

	#single neuron neural network 
	neural_network = NeuralNetwork()

	print 'Random Starting Synaptic weights'
	print neural_network.synaptic_weights

	#trainingset we have four examples , 3 input values , and one output value
	trainingset_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	trainingset_outputs = array([[0,1,1,1]]).T


	#train the neural network using a traiing set 
	#number of iterations will be 10,000 making small changes depending on the error

	neural_network.train(trainingset_inputs , trainingset_outputs , 10000)

	print 'New synaptic weights after training with neural network'
	print neural_network.synaptic_weights


	#test the neural network with a new situation
	print 'Considering new situation to test'
	print neural_network.think(array([1,0,0]))


