import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
if len(sys.argv)==1:
    print("Missing parameters! Use like:\nabc.py PSIZE LIMIT FUNC DIM LB UB ITERATIONS EXP_NUM\n\nWhere:\nPSIZE is the population size (should be an integer)\nLIMIT is the maximum amount of iterations without improvement for a bee (should be an integer)\nFUNC is the function to be optimized (should be one of the following strings without quotes: rastrigin, shaffer, griewank)\nDIM is the dimension of the function (should be an integer)\nLB is the lower bound (should be a float)\nUB is the upper bound (should be a float)\nITERATIONS is the stop criterion, the number of iterations (should be an integer)\nEXP_NUM is the number of experiments for statistical comparison (should be an integer)\n\nExample:\npython abc.py 100 200 shaffer 30 -100.0 100.0 1000 100\n")
    exit(0)
PSIZE = int(sys.argv[1])
LIMIT = int(sys.argv[2])
FUNC = sys.argv[3]
DIM = int(sys.argv[4])
LB = float(sys.argv[5])
UB = float(sys.argv[6])
ITERATIONS = int(sys.argv[7])
EXP_NUM = int(sys.argv[8])

class Bee:
    def __init__(self, dim, lb, ub, params=[], fitness='inf'):
        self.dim, self.lb, self.ub, self.fitness = dim, lb, ub, float(fitness)
        self.params = np.random.uniform(lb,ub,dim) if params==[] else params
        self.calc_fitness()
        self.stagnation_counter = 0
        self.past_fitness = self.fitness
    def calc_fitness(self):
        if self.params!=[]:
            if FUNC == 'rastrigin':
                self.fitness = 10.0*self.dim + np.sum(self.params**2 - 10.0*np.cos(2.0*np.pi*self.params)) #rastringin
            elif FUNC == 'griewank':
                self.fitness = 1 + (1./4000.)*np.sum(self.params**2) - np.prod(np.cos(self.params/np.sqrt(np.arange(self.dim)+1))) #griewank
            elif FUNC == 'shaffer':
                s = 0.0
                self.fitness = 0.0
                for i in range(self.dim - 1):
                    s = np.sqrt(self.params[i]**2 + self.params[i+1]**2)
                    self.fitness += ((1./(self.dim-1))*np.sqrt(s)*(np.sin(50.0*s**(1./5))+1.0))**2.

class Colony:
    def __init__(self, dim, lb, ub, colony_size, stagnation_limit, iterations):
        self.dim, self.lb, self.ub, self.colony_size, self.stagnation_limit, self.iterations = dim, lb, ub, colony_size, stagnation_limit, iterations
        self.employees_size = colony_size/2
        self.onlookers_size = colony_size/2
        self.scouts_size = 2
        self.ranges = [np.arange(self.employees_size) for i in range(self.employees_size)]
        self.ranges = [np.delete(self.ranges[i],i) for i in range(self.employees_size)]
        self.E = [Bee(self.dim, self.lb, self.ub) for i in range(self.employees_size)] #Employees
        self.O = [Bee(self.dim, self.lb, self.ub, params=[]) for i in range(self.onlookers_size)] #Onlookers
        self.v = Bee(self.dim, self.lb, self.ub)
        self.best_fit = Bee(self.dim, self.lb, self.ub, params=[], fitness='inf')
        self.epsilon = np.finfo(float).eps
    def foragers(self):
        k = [np.random.choice(self.ranges[i]) for i in range(self.employees_size)]
        for _i in range(self.employees_size):
            phi = np.random.uniform(-1,1,self.dim)
            j = np.random.choice(self.dim, self.dim)
            for _j in range(self.dim):
                self.v.params[_j] = self.E[_i].params[j[_j]] + phi[_j]*(self.E[_i].params[j[_j]] - self.E[k[_i]].params[j[_j]])
            self.v.calc_fitness()
            if self.v.fitness < self.E[_i].fitness:
                self.E[_i].params = np.copy(self.v.params)
                self.E[_i].fitness = self.v.fitness
                self.E[_i].past_fitness = self.v.fitness
                self.E[_i].stagnation_counter = 0
            else:
                self.E[_i].stagnation_counter += 1
            if self.E[_i].fitness < self.best_fit.fitness:
                self.best_fit.params = np.copy(self.E[_i].params)
                self.best_fit.fitness = self.E[_i].fitness
    def dance(self):
        s = sum([self.E[i].fitness for i in range(self.employees_size)])
        P = np.cumsum([self.E[i].fitness/(s + self.epsilon) for i in range(self.employees_size)])
        for i in range(self.onlookers_size):
            _rand = np.where(P > np.random.uniform(0,1))
            if _rand[0].size == 0:
                _rand = 0
            else:
                _rand = _rand[0][0]
            k = np.random.choice(self.employees_size)
            while k==_rand:
                k = np.random.choice(self.employees_size)
            phi = np.random.uniform(0,1,self.dim)
            j = np.random.choice(self.dim, self.dim)
            for _j in range(self.dim):
                self.v.params[_j] = self.E[_rand].params[j[_j]] + phi[_j]*(self.E[_rand].params[j[_j]] - self.E[k].params[j[_j]])
            self.v.calc_fitness()
            if self.v.fitness < self.E[_rand].fitness:
                self.E[_rand].params = np.copy(self.v.params)
                self.E[_rand].fitness = self.v.fitness
                self.E[_rand].past_fitness = self.v.fitness
                self.E[_rand].stagnation_counter = 0
            if self.E[_rand].fitness < self.best_fit.fitness:
                self.best_fit.params = np.copy(self.E[_rand].params)
                self.best_fit.fitness = self.E[_rand].fitness
    def scouts(self):
        for i in range(self.employees_size):
            if self.E[i].stagnation_counter > self.stagnation_limit:
                if self.E[i].fitness != self.best_fit.fitness:
                    self.E[i].params = np.random.uniform(self.lb, self.ub, self.dim)
                    self.E[i].calc_fitness()
                    if self.E[i].fitness < self.best_fit.fitness:
                        self.best_fit.params = np.copy(self.E[i].params)
                        self.best_fit.fitness = self.E[i].fitness
                    self.E[i].stagnation_counter = 0
                else:
                    self.E[i].stagnation_counter = 0
    def print_pop(self, msg):
        print(msg)
        for i, bee in enumerate(self.E):
            print('fitness '+str(i)+': '+str(bee.fitness))
F = np.zeros((EXP_NUM,ITERATIONS))
for k in range(EXP_NUM):
    P = Colony(DIM, LB, UB, PSIZE, LIMIT, ITERATIONS)
    P.E = sorted(P.E, key=lambda x: x.fitness)
    print('Run '+str(k))
    for i in range(P.iterations):
        P.foragers()
        P.dance()
        P.scouts()
        F[k][i] = min([P.E[j].fitness for j in range(P.colony_size/2)])

np.savetxt('abc_'+str(FUNC)+'_'+str(DIM)+'_dim.csv', F, delimiter=',')
plt.plot(np.arange(ITERATIONS), np.mean(F,axis=0))
plt.savefig('abc_'+str(FUNC)+'_'+str(DIM)+'_dim_mean.png')
#plt.show()
plt.plot(np.arange(ITERATIONS), np.min(F,axis=0))
plt.savefig('abc_'+str(FUNC)+'_'+str(DIM)+'_dim_min.png')

#F = np.min(F, axis=0)
#plt.plot(np.arange(P.iterations),F)
#plt.savefig('abc_griewank.png')
#plt.show()
