from numpy import *
import numpy as np
import math

class NeuralNetwork():
    def __init__(self):
        #seed random num generator, so generates same numbers every time program runs
        random.seed(1)

        #model single neuron with 3 input connections and 1 output connection
        #assign random weight to a 3 x 1 matrix, with values in range -1 to 1 and mean 0
        #random.rand works exactly to random.random
        #more info on numpy random.random go here: http://memobio2015.u-strasbg.fr/conference/FICHIERS/Documentation/doc-numpy-html/reference/generated/numpy.random.random.html
        self.synapticWeights = 2 * random.rand(3,1) - 1

    #sigmoid func; S shaped curve
    #pass weighted sum of inputs thru this func to normalize them between 0 and 1
    #exp(-x) is equal to the e^-x
    #x is the summation of the weighted inputs
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #derivative of sigmoid func
    #gradient of the sigmoid curve
    #indicates how confident we are about the existing weight
    #for info about how the derivative is done visit: https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    #you can also use derivative-calculator.net
    def sigmoidDerivative(self, x):
        return x * (1-x)

    #the neural network "thinking"
    def think(self, inputs):
        #pass inputs thru neural network (the single neuron)
        #remember that the dot product is the sum of the products of corresponding elements of two vectors
        #ex. dot product of [1, 2, 3] with [4, 5, 6] = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        return self.sigmoid(dot(inputs, self.synapticWeights))

    #train the neural network thru process of trial and error
    #adjusting synaptic weights each time
    def train(self, trainingSetInputs, trainingSetOutputs, numTrainingIterations):
        #might consider using xrange() instead of range() since xrange uses less memory and is faster since it only evaluates values that are required by lazy evaluation
        #however, it is deprecated in Python 3
        #for more info about range vs xrange, visit: https://www.geeksforgeeks.org/range-vs-xrange-python/
        #for iteration in xrange(numTrainingIterations):
        for iteration in range(numTrainingIterations):
            #pass training set thru neural network (a single neuron)
            output = self.think(trainingSetInputs)

            #calculate the error (diff between desired output and the predicted output)
            error = trainingSetOutputs - output

            #multiply the error by the input and again by the gradient of the sigmoid curve
            #less confident weights are adjusted more
            #inputs that are zero do not cause changes to the weights
            #the .T function transposes input matrix from hori to vert
            adjustment = dot(trainingSetInputs.T, error * self.sigmoidDerivative(output))

            #adjust the weights
            self.synapticWeights += adjustment
        
if __name__ == "__main__":

    #initialize single neural network
    neuralNet = NeuralNetwork()

    #goal of the neural network = make the output be equal to 1 if the middle number of the input array is a 1
    #other return a 0
    #ex. an array like [0, 1, 0] should result in an output of 1 while an array like [0, 0, 1] should result in 
    #an output of 0

    print("Starting synaptic weights: ")
    print(neuralNet.synapticWeights) 
    '''
    #training set = 5 examples, each that have 3 input values and 1 output value
    #these sets 
    trainingSetInputs = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
    trainingSetOutputs = np.array([[1, 1, 0, 0, 1]]).T
    '''
    #training set = 6 examples, each that have 3 input values and 1 output value
    trainingSetInputs = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 0, 1]])
    trainingSetOutputs = np.array([[1, 1, 0, 0, 1, 0]]).T
    
    #train neural net using training set
    #repeat process n times and make adjustments each time (back propagation though in a very small scale since
    #there's only one layer and only one neuron 
    neuralNet.train(trainingSetInputs, trainingSetOutputs, 100000)

    print("Final synaptic weights: ")
    print(neuralNet.synapticWeights)

    #testing the neural net with new inputs
    t1 = [1, 0, 0]
    t2 = [0, 0, 0]
    t3 = [1, 1, 0]

    print("Test #1 " + str(t1) + " -> ?: ")
    print(neuralNet.think(np.array(t1)))

    print("Test #2 " + str(t2) + " -> ?: ")
    print(neuralNet.think(np.array(t2)))

    print("Test #3 " + str(t3) + " -> ?: ")
    print(neuralNet.think(np.array(t3)))