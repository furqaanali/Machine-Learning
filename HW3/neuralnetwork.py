import numpy as np
np.random.seed(100)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:

    def __init__(self,x=[[]],y=[],numLayers=2,numNodes=2,eta=0.001,maxIter=10000):
        self.data = np.append(x, np.ones([len(x), 1]), 1)
        self.labels = np.array(y)
        self.nLayers = numLayers - 1                                #nLayers will hold the number of hidden layers (1 is subtracted to exclude output layer)
        self.nNodes = numNodes
        self.eta = eta * 100            # increased learning rate to allow fewer iterations before nearing convergence
        self.maxIt = maxIter // 10      # decreased iterations to allow faster runtime


        if self.nLayers == 0:                                       # no hidden layers
            self.weights = [np.random.rand(len(x[0]) + 1, 1)]

        else:
            self.weights = [np.random.rand(len(x[0]) + 1, self.nNodes)] #create the weights from the inputs to the first layer          +1 for bias node

            for i in range( self.nLayers - 1):                         # numLayers - 1 is the number of connections between each hidden layer
                self.weights.append(np.random.rand(self.nNodes + 1, self.nNodes)) #create the random weights between internal layers       +1 for bias node

            self.weights.append(np.random.rand(self.nNodes + 1, 1)) #create weights from final layer to output node                     +1 for bias node

        self.outputs = []
        for i in range( self.nLayers + 1):
            self.outputs.append([])

        for i in range( self.maxIt):
            self.train()


    def train(self):
        for index in range( len( self.data)):
            self.feedforward( self.data[ index])
            self.backprop( self.data[ index], self.labels[ index])


    def predict(self,x=[]):
        self.feedforward(np.append(x, 1))
        return self.outputs[-1][0]


    def feedforward(self, point):
        self.outputs[0] = sigmoid( np.dot( point, self.weights[ 0]))
        self.outputs[0] = np.append( self.outputs[ 0], 1)

        for index in range( 1, len( self.outputs)):
            self.outputs[ index] = sigmoid( np.dot( self.outputs[ index - 1], self.weights[ index]))

            if index != len( self.outputs) - 1:                             # output layer should not have a bias node
                self.outputs[ index] = np.append( self.outputs[ index], 1)


    def backprop(self, point, label):
        sensitivities = []
        for i in range( self.nLayers + 1):
            sensitivities.append([])

        sensitivities[ -1] = ( label - self.outputs[ -1]) * sigmoid_derivative( self.outputs[ -1])
        for index in range( len( sensitivities) - 2, -1, -1):
            sensitivities[ index] = ( np.dot( sensitivities[ index + 1], self.weights[ index + 1].T) * sigmoid_derivative(self.outputs[ index]))[:-1]

        changes = []
        for i in range(self.nLayers + 1):
            changes.append([])

        for index in range(len( changes) - 1, 0, -1):
            changes[ index] = np.outer( self.outputs[ index - 1], sensitivities[ index])

        changes[0] = np.outer( point, sensitivities[ 0])

        for index in range( len( self.weights)):
            self.weights[ index] += self.eta * changes[ index]
