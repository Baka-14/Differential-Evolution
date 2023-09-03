
import random 
import numpy as np
from matplotlib import pyplot as plt
import math 

# Population size: 20, 50, 100, 200.
# Num of Gens: 50, 100, 200.
#Parameters to change-: 
populationSize=200
genNums=200
Cp=0.80 
K=0.5 
boundariesEgg=[-512,512]
boundariesHolder=[-10,10]

#Object function value vs Value on "i"th generation
#Plot Graph obj func wrt avg value each generation 
#Plot Gpaph obj func wrt best solution each generation 

#need to find global minimum value of the functions 
def eggHolder(x,y):
    #here x and y lie between -512 and +512 
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(y + x/2 + 47)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2

def holderTable(x, y):
    pi = np.pi
    term1 = -abs(np.sin(x) * np.cos(y))
    term2 = np.exp(abs(1 - np.sqrt(x**2 + y**2) / pi))
    return term1 * term2

def initialisePopulation(populationSize,boundaries):
    #initialising population as np array for faster calculations
    population=np.empty((populationSize,2))

    for i in range(0,populationSize):
        x_value = random.uniform(boundaries[0], boundaries[1])
        y_value = random.uniform(boundaries[0], boundaries[1])
        population[i]=[x_value,y_value]

    return population

#intialising intial population
population=initialisePopulation(populationSize,boundariesHolder)


def randomIndexs(numberRequired):
    indexNo=[]
    while len(indexNo)<numberRequired:
        randomIndex=random.randint(0,populationSize-1)
        if randomIndex not in indexNo:
            indexNo.append(randomIndex)

    return indexNo

def genmutantVector(population, F,boundaries):
    while True:
        randomIndex = randomIndexs(4)
        i, r1, r2, r3 = randomIndex
        Xi, Xr1, Xr2, Xr3 = population[i], population[r1], population[r2], population[r3]
        Xi = Xi + K * (Xr1 - Xi) + F * (Xr2 - Xr3)
        # Check if Xi is within the specified constraints
        if boundaries[0] <= Xi[0] <= boundaries[1] and boundaries[0] <= Xi[1] <= boundaries[1]:
            return Xi


def elitism(old,new,function):
    oldValue=function(old[0],old[1])
    newValue=function(new[0],new[1])
    if oldValue>newValue:
        return new,newValue
    else:
        return old,oldValue

def generateTrail(mutantVector,parentVector):
    n=len(mutantVector)
    trialVector=np.empty(n)
    for i in range(0,n):
        npr=random.uniform(0,1)
        if npr>Cp:
            trialVector[i]=mutantVector[i]
        else:
            trialVector[i]=parentVector[i]
    return trialVector

def plotGraph(x,y,title,nameofXaxis,nameofYaxis):
    plt.figure(num=f'{title}')
    plt.title(f'Population Size:{populationSize} and No of Generations:{genNums}')
    plt.ylabel(f'{nameofYaxis}')
    plt.xlabel(f'{nameofXaxis}')
    plt.plot(x,y)
    plt.show()

def average(values):
    sum=0
    for i in range(0,len(values)):
        sum=sum+values[i]
    return sum/len(values)


def differentialEvolution(genNums,populationSize,population,function):
    globalBest=np.empty(genNums)
    avg=np.empty(genNums)
    values=np.empty(populationSize)
    xaxis=[]
    #iterating through each generation
    for gen in range(0,genNums):
        F=random.uniform(-2, 2) #change each generation 
        mutantVector=genmutantVector(population,F,boundariesHolder)
        for candidate in range(0,populationSize):
            trailVector=generateTrail(mutantVector,population[candidate])
            population[candidate],values[candidate]=elitism(population[candidate],trailVector,function)
        #storing the values
        avg[gen]=average(values)
        globalBest[gen]=np.min(values)
        xaxis.append(gen)
    return avg,globalBest,xaxis

# avg,globalBest,xaxis=differentialEvolution(genNums,populationSize,population,eggHolder)

avg,globalBest,xaxis=differentialEvolution(genNums,populationSize,population,holderTable)
print(globalBest[genNums-1])
#plotting average solutions 
plotGraph(xaxis,avg,"Average Values vs Generation","Generation Number","Avg Value")

#plotting best solutions
plotGraph(xaxis,globalBest,"Global Best Values vs Generation","Generation Number","Global Best")


#ans is -959.6407
#ans is -19.2085