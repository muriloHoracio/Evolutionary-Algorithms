import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
if len(sys.argv)==1:
    print("Missing parameters! Use like:\ngwo.py PSIZE FUNC DIM LB UB ITERATIONS EXP_NUM\n\nWhere:\nPSIZE is the population size (should be an integer)\nFUNC is the function to be optimized (should be one of the following strings without quotes: rastrigin, shaffer, griewank)\nDIM is the dimension of the function (should be an integer)\nLB is the lower bound (should be a float)\nUB is the upper bound (should be a float)\nITERATIONS is the stop criterion, the number of iterations (should be an integer)\nEXP_NUM is the number of experiments for statistical comparison (should be an integer)\n\nExample:\npython gwo.py 10 shaffer 30 -100.0 100.0 1000 100\n")
    exit(0)
PSIZE = int(sys.argv[1])
FUNC = sys.argv[2]
DIM = int(sys.argv[3])
LB = float(sys.argv[4])
UB = float(sys.argv[5])
ITERATIONS = int(sys.argv[6])
EXP_NUM = int(sys.argv[7])

norml = 1.0/(DIM-1)

def calc_fitness(params):
    fitness = 0.0
    if FUNC == 'rastrigin': # rastrigin
        fitness = 10.0*DIM
        for i in range(DIM): 
            fitness += math.pow(params[i],2) - 10.0*math.cos(2.0*math.pi*params[i])
    elif FUNC == 'shaffer': #shaffer
        fitness = 0.0
        for i in range(DIM-1):
            s = math.sqrt(math.pow(params[i],2.) + math.pow(params[i+1],2.))
            fitness += math.pow(norml*math.sqrt(s)*(math.sin(50.0*math.pow(s,0.2))+1),2.)
    elif FUNC == 'griewank': #griewank
        s = 0.0
        p = 1.0
        fitness = 1.0
        for i in range(DIM):
            s += math.pow(params[i],2)
            p *= math.cos(params[i]/math.sqrt(i+1))
        fitness += 0.00025*s - p
    return fitness

class Wolf:
    def __init__(self, params=[],fitness='inf'):
        self.params = np.random.uniform(LB,UB,DIM) if params==[] else params
        self.fitness = calc_fitness(self.params)
    def encircle(self, alpha, beta, delta, a, A, C):
        D_alpha = np.abs(C[0]*alpha - self.params)
        D_beta = np.abs(C[1]*beta - self.params)
        D_delta = np.abs(C[2]*delta - self.params)
        X_1 = alpha - A[0]*(D_alpha)
        X_2 = beta - A[1]*(D_beta)
        X_3 = delta - A[2]*(D_delta)
        self.params = ((X_1 + X_2 + X_3)/3.0)[0]
    def update_wolf(self, params, fitness):
        self.params = np.copy(params)
        self.fitness = fitness
class Population:
    def __init__(self):
        self.iter = 0
        self.P = sorted([Wolf() for i in range(PSIZE)],key=lambda x: x.fitness)
        self.a = np.ones(DIM)*2
        self.A = [[2.*self.a*np.random.uniform(0,1,DIM) - self.a] for i in range(3)]
        self.C = [[2.*np.random.uniform(0,1,DIM)] for i in range(3)]
        self.a_decay = np.ones(DIM)*(2.0/ITERATIONS)
        self.alpha = Wolf(params=self.P[0].params, fitness=self.P[0].fitness)
        self.beta  = Wolf(params=self.P[1].params, fitness=self.P[1].fitness)
        self.delta = Wolf(params=self.P[2].params, fitness=self.P[2].fitness)
    def step(self):
        for i in self.P:
            i.encircle(self.alpha.params, self.beta.params, self.delta.params, self.a, self.A, self.C)
    def update_coeficients(self):
        self.a = self.a - self.a_decay
        self.A = [[2.*self.a*np.random.uniform(0,1,DIM) - self.a] for i in range(3)]
        self.C = [[2.*np.random.uniform(0,1,DIM)] for i in range(3)]
    def evaluate_pop(self):
        for i in self.P:
            i.fitness = calc_fitness(i.params)
        self.P = sorted(self.P,key=lambda x: x.fitness)
#        if self.P[0].fitness < self.alpha.fitness:
        self.alpha.params = np.copy(self.P[0].params)
        self.alpha.fitness = self.P[0].fitness
#        if self.P[1].fitness < self.beta.fitness:
        self.beta.params = np.copy(self.P[1].params)
        self.beta.fitness = self.P[1].fitness
#        if self.P[2].fitness < self.delta.fitness:
        self.delta.params = np.copy(self.P[2].params)
        self.delta.fitness = self.P[2].fitness
        self.iter += 1
    def print_pop(self, msg):
        print(msg)
        for i in range(PSIZE):
            print('fitness: ', self.P[i].fitness)
F = np.zeros((EXP_NUM, ITERATIONS))
for j in range(EXP_NUM):
    print('Run ',str(j))
    P = Population()
    best = Wolf(params=P.P[0].params)
    for i in range(ITERATIONS):
        P.step()
        P.update_coeficients()
        P.evaluate_pop()
        if P.alpha.fitness < best.fitness:
            best.fitness = P.alpha.fitness
            best.params = np.copy(P.alpha.params)
        F[j][i] = best.fitness
np.savetxt('gwo_'+str(FUNC)+'_'+str(DIM)+'_dim.csv', F, delimiter=',')
plt.plot(np.arange(ITERATIONS), np.mean(F,axis=0))
plt.savefig('gwo_'+str(FUNC)+'_'+str(DIM)+'_dim_mean.png')
#plt.show()
plt.plot(np.arange(ITERATIONS), np.min(F,axis=0))
plt.savefig('gwo_'+str(FUNC)+'_'+str(DIM)+'_dim_min.png')
#plt.show()
#print('argmin', np.argmin(P.F_best), P.F_best[np.argmin(P.F_best)])
#plt.plot(np.arange(P.iterations),P.F_best)
#plt.show()
#plt.plot(np.arange(P.iterations),P.F_avg)
#plt.show()
