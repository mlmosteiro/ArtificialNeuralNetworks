import random
from math import exp

import sys
from pip._vendor.distlib.compat import raw_input #Keyboard input

numNeurons =[]
desiredAnswers=[]
patterns=[]


class SeedNeuron(object):
    def __init__(self, id, inputs, neuronCurrentPattern):
        self.id = id
        self.inputs= inputs
        self.answer = self.inputs[neuronCurrentPattern]

class Neuron(object):
    def __init__(self, id):
        super().__init__()
        self.id = id
        self.learningRate = 0.1
        self.inputNeurons = {} #Neurons that provides the inputs , It is a dictionary of the form {Neuron,weight}
        self.outputNeurons = [] #Neurons that I provide the input (they receive my output)
        #self.weights = []
        self.answer = 0
        self.error = 0

    def addInputNeuron(self, targetNeuron):
        self.inputNeurons[targetNeuron]= self.newWeight()

    def addOutputNeuron(self, targetNeuron):
        self.outputNeurons.append(targetNeuron)

    def newWeight(self):
        return random.random()

    def calculateSin(self):
        tmp = 0
        for neuron in self.inputNeurons:
            tmp += neuron.answer * self.inputNeurons[neuron]
        self.answer = 2 / (1 + exp(-tmp)) - 1

    def getError(self, desiredAnswer):
        if self.outputNeurons: #Hidden neuron
            tmp =0
            for outNeuron in self.outputNeurons:
                tmp += outNeuron.error * outNeuron.inputNeurons[self]
            self.error = tmp *  (1 - (self.answer * self.answer)) / 2

        else: #Output neuron
            self.error = (desiredAnswer- self.answer) * (1 - (self.answer * self.answer)) / 2
        return self.error

    def adjustWeights(self):
        for neuron in self.inputNeurons: #for each input, we have a weight that we want to adjust
            self.inputNeurons[neuron] += (self.learningRate*self.error*neuron.answer)

class NeuralNetwork(object):

    def __init__(self, numNeurons, patterns, desiredAnswers):
        self.numNeurons = numNeurons
        self.patterns = patterns
        self.numPatterns = len(self.patterns)
        self.numInputs = len(self.patterns[0])
        self.currentPattern = 0
        self.currentInput = 0
        self.desiredAnswers = desiredAnswers
        self.neuralNetwork =[]

        for noLayer in range(len(numNeurons)): # Num of layers ( including the seed layer)
            layer = []
            for i in range(numNeurons[noLayer]):  # Num of new neurons in that layer
                if(noLayer==0):
                    newNeuron = SeedNeuron(str(noLayer)+"."+str(i), patterns[i], self.currentPattern)
                else:
                    newNeuron = Neuron(str(noLayer)+"."+str(i))
                layer.append(newNeuron)
            self.neuralNetwork.append(layer)
        self.createConections()

    def createConections(self):
        #Output conections
        numLayer = 1 #Do not take in count seedLayer
        for noLayer in range(1,len(numNeurons)):
            if (numLayer < len(self.neuralNetwork)-1):
                for neuron in self.neuralNetwork[numLayer]:
                    for targetNeuron in self.neuralNetwork[numLayer+1]:
                        neuron.addOutputNeuron(targetNeuron)
                numLayer+=1

        # Input conections
        numLayer = len(self.neuralNetwork) - 1
        for noLayer in range(len(numNeurons)-1, -1, -1):
            if (numLayer > 0):
                for neuron in self.neuralNetwork[numLayer]:
                    for targetNeuron in self.neuralNetwork[numLayer - 1]:
                        neuron.addInputNeuron(targetNeuron)
                numLayer -= 1

    def forwardPropagation(self):
        for layer in self.neuralNetwork:
            for neuron in layer:  # Forward propagation of the input
                if type(neuron) is not SeedNeuron:
                    neuron.calculateSin()

    def backwardPropagation(self):
        for noLayer in range(len(numNeurons) - 1, -1, -1):  # Backward propagation of the error
            if (noLayer > 0):
                for neuron in self.neuralNetwork[noLayer]:
                    neuron.getError(self.desiredAnswers[self.currentInput])

    def adjustWeights(self):
        for layer in self.neuralNetwork:
            for neuron in layer:
                if type(neuron) is not SeedNeuron:
                    neuron.adjustWeights()

    def printStadistics(self):
        for layer in self.neuralNetwork:
            for neuron in layer:
                if(type(neuron) is not SeedNeuron):
                    print("Neuron: " , neuron.id ,
                          "\t Answer: " , neuron.answer ,
                          "\t Desidered answer: " , self.desiredAnswers[self.currentInput],
                          "\t Error: ",neuron.error)

    def printConections(self):
        noLayer = 0
        noNeuron = 0
        for layer in self.neuralNetwork:
            print("Capa " , noLayer , "\n")
            for neuron in layer:
                if type(neuron) is not SeedNeuron:
                    print("\tNeurona " , noNeuron , "\n\tConectada INPUT con:")
                    for conection in neuron.inputNeurons:
                        print(conection.id)
                    print("\n\tConectada OUTPUT con:")
                    for conection in neuron.outputNeurons:
                        print(conection.id)
                    noNeuron+=1
            noLayer+=1


def readFile():
    with open(sys.argv[1], "r") as inputFile:
        line = inputFile.readline()
        constants = line.split()
        numPatterns = int(constants[0])             #Number of patterns
        numInputs = int(constants[1])               #Number of inputs per pattern

        line = inputFile.readline()
        numNeurons = [int(i) for i in line.split()]              #Array with the number of neurons per layer

        line = inputFile.readline()
        desiredAnswers = [float(i) for i in line.split()]        #Array with the desired answers for each pattern

        for i in range (numPatterns):
            line = inputFile.readline()
            inputs =[float(i) for i in line.split()]
            patterns.append(inputs)

if __name__ == '__main__':

    numNeurons = [1, 1, 3, 1]
    patterns = [[0.0, 90.0, 180.0, 360.0]]
    desiredAnswers = [0, 1, 0, -1]

    #readFile()
    neuralNetwork = NeuralNetwork(numNeurons, patterns, desiredAnswers)

    iteration = 0
    while True:
        print("------Iteration ", iteration, "------")
        print("------Value: ", patterns[neuralNetwork.currentPattern][neuralNetwork.currentInput],"------")

        neuralNetwork.forwardPropagation()                          # Forward propagation of the inputs
        neuralNetwork.backwardPropagation()                         # Backward propagation of the error
        neuralNetwork.printStadistics()
        neuralNetwork.adjustWeights()


        neuralNetwork.currentInput = (neuralNetwork.currentInput + 1)%neuralNetwork.numInputs
        if(neuralNetwork.currentInput == 0):
            neuralNetwork.currentPattern = (neuralNetwork.currentPattern + 1) % neuralNetwork.numPatterns
        iteration+=1

        print("Press any key to continue, X to finish")
        k = raw_input('> ')
        if k == 'x' or k=='X':
            break


