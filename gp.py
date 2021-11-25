# GP assignment

import numpy as np
import pandas as pd
import math
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
from collections import defaultdict

# sphere function
def sphere(x):
    return sum(np.square(x))

# Rastrigin function
def rastrigin(x):
    return 10*len(x) + sum(np.square(x)-10*np.cos((2*np.pi*x)))

class Particle:

    def __init__(self, func, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_particle_pos = self.position
        self.dim = dim
        self.func = func

        self.fitness = func(self.position)
        self.best_particle_fitness = self.fitness  # we couldd start with very large number here,
        # but the actual value is better in case we are lucky

    def setPos(self, pos):
        self.position = pos
        self.fitness = self.func(self.position)
        if self.fitness < self.best_particle_fitness:  # to update the personal best both
            # position (for velocity update) and
            # fitness (the new standard) are needed
            # global best is update on swarm leven
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, best_self_pos, best_swarm_pos, a3=None, Z=None, repulse=False):
        # Here we use the canonical version
        # V <- inertia*V + a1r1 (peronal_best - current_pos) + a2r2 (global_best - current_pos)
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size=self.dim)
        r2 = np.random.uniform(low=0, high=1, size=self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
        # the next line is the main equation, namely the velocity update,
        # the velocities are added to the positions at swarm level
        if repulse:
            new_vel = inertia * cur_vel + np.multiply(a1r1, best_self_dif) + np.multiply(a2r2, best_swarm_dif) + a3 * Z
        else:
            new_vel = inertia * cur_vel + np.multiply(a1r1, best_self_dif) + np.multiply(a2r2, best_swarm_dif)
        self.velocity = new_vel
        return new_vel

class PSO:

    def __init__(self, w, a1, a2, population_size, time_steps, search_range, dim, func, repulse=False,
                 error_bound=0.0001, repeated=50, p=0.00001, a3=None, print_steps=False):

        # Here we use values that are (somewhat) known to be good
        # There are no "best" parameters (No Free Lunch), so try using different ones
        # There are several papers online which discuss various different tunings of a1 and a2
        # for different types of problems
        self.w = w  # Inertia
        self.a1 = a1  # Attraction to personal best
        self.a2 = a2  # Attraction to global best
        self.a3 = a3
        self.dim = dim
        self.func = func
        self.p = p
        self.no_term = True
        self.S = 0
        self.search_range = search_range
        self.repulse = repulse
        self.error_bound = error_bound
        self.repeated = repeated
        self.swarm = [Particle(func, dim, -search_range, search_range) for i in range(population_size)]
        self.time_steps = time_steps
        self.print_steps = print_steps
        self.best_swarm_fitness_step = 0
        self.stopping_fitness = 0
        self.stopping_fitness_step = 0
        print('init')

        # Initialising global best, you can wait until the end of the first time step
        # but creating a random initial best and fitness which is very high will mean you
        # do not have to write an if statement for the one off case
        self.best_swarm_pos = np.random.uniform(low=-500, high=500, size=dim)
        self.best_swarm_fitness = 1e100
        self.best_swarm_fitness_record = []
        self.delta = []
        self.delta_all = []
        self.divergent = False

    def run(self):
        for t in range(self.time_steps):
            for p in range(len(self.swarm)):
                particle = self.swarm[p]

                if self.repulse:
                    repulse_terms = []
                    r3 = np.random.uniform(low=0, high=1, size=self.dim)
                    all_other = deepcopy(self.swarm)
                    del all_other[p]
                    for k in all_other:
                        neighbour = np.subtract(k.position, particle.position)
                        x = (np.multiply(r3, neighbour)) / ((np.linalg.norm(neighbour)) ** 2)
                        repulse_terms.append(x)
                    Z = -sum(repulse_terms)
                else:
                    Z = None

                new_veloctiy =  particle.updateVel(self.w, self.a1, self.a2,
                                                                      particle.best_particle_pos, self.best_swarm_pos,
                                                                      self.a3, Z, repulse=self.repulse)

                new_position = particle.position + new_veloctiy

                if new_position @ new_position > 1.0e+18:  # The search will be terminated if the distance
                    # of any particle from center is too large
                    #print('Time:', t, 'Best Pos:', self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                    self.divergent = True

                self.swarm[p].setPos(new_position)
                new_fitness = self.func(new_position)

                if new_fitness < self.best_swarm_fitness:  # to update the global best both
                    # position (for velocity update) and
                    # fitness (the new group norm) are needed
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position

            self.delta = [abs(self.best_swarm_fitness - i) for i in self.best_swarm_fitness_record]
            self.best_swarm_fitness_record.append(self.best_swarm_fitness)

            self.S = 0
            for i in range(len(self.delta) - 1, 0, -1):
                if self.delta[i] <= self.error_bound:
                    self.S += 1
                else:
                    break

            if self.S == self.repeated and self.no_term:
                print('Time:', t, 'Best Fit:', self.best_swarm_fitness, 'Best Pos:', self.best_swarm_pos)
                self.no_term = False
                self.stopping_fitness = self.best_swarm_fitness
                self.stopping_fitness_step = t
                # raise SystemExit('Convergence: Termination condition satisfied')

            if self.print_steps:
                if t % 100 == 0:  # we print only two components even it search space is high-dimensional
                    print("Time: %6d,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (
                        t, self.best_swarm_fitness, self.best_swarm_pos[0], self.best_swarm_pos[1]), end=" ")
                    if self.dim > 2:
                        print('...')
                    else:
                        print('')

        self.delta_all = [abs(self.best_swarm_fitness - i) for i in self.best_swarm_fitness_record]
        self.S = 0
        for i in range(len(self.delta_all) - 1, 0, -1):
            if self.delta_all[i] <= self.error_bound:
                self.S += 1
            else:
                break
        self.best_swarm_fitness_step = self.time_steps - self.S

        if self.divergent:
            print("Most likely divergent.")


params = {"w=0.9,a1=a2=1": {"w": 0.9, "a1": 2, "a2": 2, "r": 5},#sphere=5  #rastrigin=None
          "w=0.7,a1=a2=1.4": {"w": 0.7, "a1": 1.4, "a2": 1.4, "r": 6},#sphere=6  #rastrigin=15
          "w=0.7,a1=a2=2": {"w": 0.7, "a1": 2, "a2": 2, "r": 30}, #sphere=30  #rastrigin=250
          "w=0.7,a1=1.6,a2=0.6": {"w": 0.7, "a1": 1.6, "a2": 0.6, "r": 3}, #sphere=3  #rastrigin=250
          "w=0.5,a1=a2=2": {"w": 0.5, "a1": 2, "a2": 2, "r": 8}, #sphere=8  #rastrigin=100
          "w=0.3,a1=a2=2": {"w": 0.3, "a1": 2, "a2": 2, "r": 4}, #sphere=4  #rastrigin=100
          "w=0.1,a1=a2=2": {"w": 0.1, "a1": 2, "a2": 2, "r": 3}}#sphere=3  #rastrigin=100

params = {"w=0.9,a1=a2=1": {"w": 0.9, "a1": 2, "a2": 2, "r": 5},#sphere=5  #rastrigin=None
          "w=0.7,a1=a2=1.4": {"w": 0.7, "a1": 1.4, "a2": 1.4, "r": 15},#sphere=6  #rastrigin=15
          "w=0.7,a1=a2=2": {"w": 0.7, "a1": 2, "a2": 2, "r": 250}, #sphere=30  #rastrigin=250
          "w=0.7,a1=1.6,a2=0.6": {"w": 0.7, "a1": 1.6, "a2": 0.6, "r": 250}, #sphere=3  #rastrigin=250
          "w=0.5,a1=a2=2": {"w": 0.5, "a1": 2, "a2": 2, "r": 100}, #sphere=8  #rastrigin=100
          "w=0.3,a1=a2=2": {"w": 0.3, "a1": 2, "a2": 2, "r": 100}, #sphere=4  #rastrigin=100
          "w=0.1,a1=a2=2": {"w": 0.1, "a1": 2, "a2": 2, "r": 100}}#sphere=3  #rastrigin=100

data = defaultdict(list)
data_params = defaultdict(list)
averages = defaultdict(dict)
results = dict()
simulations = 30
for i in params.keys():
    w = params[i]["w"]
    a1 = params[i]["a1"]
    a2 = params[i]["a2"]
    r = params[i]["r"]
    print("Parameters: w= " + str(w) + ", a1= " + str(a1) + " ,a2= ", str(a2))
    print("=================================================================")
    for j in range(simulations):
        print("Running simulation " + str(j + 1))
        print("--------------------------------------------")
        x = PSO(w=w, a1=a1, a2=a2, population_size=60, time_steps=5001, search_range=5.12, dim=10,
                func=rastrigin, repulse=True, error_bound=0.0001, repeated=r, a3=2, print_steps=True)
        x.run()
        results[j] = {"stopping fitness": x.stopping_fitness,
                      "best swarm step": x.best_swarm_fitness_step,
                      "difference in fitness": abs(x.stopping_fitness - x.best_swarm_fitness),
                      "difference in steps": x.best_swarm_fitness_step - x.stopping_fitness_step,
                      "best swarm fitnesses": x.best_swarm_fitness_record}
    averages[i]["difference in fitness"] = np.average([results[k]["difference in fitness"] for k in range(simulations)])
    averages[i]["difference in steps"] = np.average([results[k]["difference in steps"] for k in range(simulations)])
    averages[i]["best swarm fitnesses"] = np.average(np.array([results[k]["best swarm fitnesses"]
                                                               for k in range(simulations)]), axis=0)
    averages[i]["best swarm step"] = np.average([results[k]["best swarm step"] for k in range(simulations)])

    data[i] = averages[i]["best swarm fitnesses"]

df = pd.DataFrame(data=data)
df.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/data_1.1_sphere.csv")
df.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/data_1.1_rastrigin.csv")
df.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/data_1.3_sphere.csv")
df.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/data_1.3_rastrigin.csv")

for i in averages.keys():
    print(str(i) + " (difference in fitness)" + ": " + str(averages[i]["difference in fitness"]))
    print(str(i) + " (difference in steps)" + ": " + str(averages[i]["difference in steps"]))
    print(str(i) + " (best swarm step)" + ": " + str(averages[i]["best swarm step"]))
    print(str(i) + " (best swarm fitness)" + ": " + str(np.average(averages[i]["best swarm fitnesses"])))
    print(" ")


particle_dim = [(30, 5), (30, 10), (30, 20),
                (60, 5), (60, 10), (60, 20),
                (90, 5), (90, 10), (90, 20),
                (120, 5), (120, 10), (120, 20)]
data = defaultdict(list)
averages = defaultdict(dict)
results = dict()
simulations = 30
for pd in particle_dim:
    particle_size = pd[0]
    dimension = pd[1]
    for j in range(simulations):
        print("Running simulation " + str(j + 1))
        print("--------------------------------------------")
        x = PSO(w=0.7, a1=2, a2=2, population_size=particle_size, time_steps=8001, search_range=5.12, dim=dimension,
                func=rastrigin, repulse=False, error_bound=0.0001, repeated=10, a3=2, print_steps=True)
        x.run()
        results[j] = {"best swarm step": x.best_swarm_fitness_step,
                      "best swarm fitnesses": x.best_swarm_fitness_record}
    averages[pd]["best swarm step"] = np.average([results[k]["best swarm step"] for k in range(simulations)])
    averages[pd]["best swarm fitnesses"] = np.average(np.array([results[k]["best swarm fitnesses"]
                                                               for k in range(simulations)]), axis=0)

    data[pd] = averages[pd]["best swarm fitnesses"]


df_pd = pd.DataFrame(data=data)
df_pd.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/data_1.2_sphere.csv")
df_pd.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/data_1.2_rastrigin.csv")

for i in averages.keys():
    print(str(i) + " (best swarm step)" + ": " + str(averages[i]["best swarm step"]))
    print(str(i) + " (best swarm fitness)" + ": " + str(np.average(averages[i]["best swarm fitnesses"])))
    print(" ")


POP_SIZE        = 100  # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 7    # maximal initial random tree depth
GENERATIONS     = 300  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 7    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate
PROB_MUTATION   = 0.2  # per-node mutation probability

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def cos(x): return math.cos(x)

#FUNCTIONS = [add, mul]
#TERMINALS = ['x1', 'x2']
FUNCTIONS = [add, sub, mul, cos]
TERMINALS = ['x1', 'x2', -10, 2, np.pi, 10]

def generate_dataset(target_func): # generate 101 data points from target_func
    dataset = []
    input_values = []
    for x in range(-100,101,2):
        x /= 19.53125
        input_values.append(x)
    rand_ind1 = [randint(0, len(input_values)-1) for i in range(len(input_values))]
    rand_ind2 = [randint(0, len(input_values)-1) for i in range(len(input_values))]
    for i in range(0, len(input_values)):
        x1 = input_values[rand_ind1[i]]
        x2 = input_values[rand_ind2[i]]
        dataset.append([x1, x2, target_func(np.array([x1, x2]))])
    return dataset


class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def node_label(self):  # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix=""):  # textual printout
        print("%s%s" % (prefix, self.node_label()))
        if self.left:  self.left.print_tree(prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x1, x2):
        if (self.data in FUNCTIONS and self.data != cos):
            return self.data(self.left.compute_tree(x1, x2), self.right.compute_tree(x1, x2))
        elif self.data == cos:
            if self.left is None:
                return self.data(self.right.compute_tree(x1, x2))
            else:
                return self.data(self.left.compute_tree(x1, x2))
        elif self.data == 'x1':
            return x1
        elif self.data == 'x2':
            return x2
        else:
            return self.data

    def random_tree(self, grow, max_depth, depth=0):  # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        if self.data in FUNCTIONS and self.data != cos:
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth=depth + 1)
        elif self.data == cos:
            if random() > 0.5:
                self.left = GPTree()
                self.left.random_tree(grow, max_depth, depth=depth + 1)
            else:
                self.right = GPTree()
                self.right.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION:  # mutate at this node
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    def size(self):  # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size() if self.left else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):  # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t

    def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other):  # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None)  # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second)  # 2nd subtree "glued" inside 1st tree

def fitness(individual, dataset): # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(individual.compute_tree(ds[0], ds[1]) - ds[2]) for ds in dataset]))

def selection(population, fitnesses, double=False, parsimony=1.5): # select one individual using tournament selection
    if double:
        winner1 = selection(population, fitnesses, double=False)
        winner2 = selection(population, fitnesses, double=False)
        size = [winner1.size(), winner2.size()]
        if random() > parsimony / 2:
            if size.index(max(size)) == 0:
                return winner1
            elif size.index(max(size)) == 1:
                return winner2
        else:
            if size.index(min(size)) == 0:
                return winner1
            elif size.index(min(size)) == 1:
                return winner2
    else:
        tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
        tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
        return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])

def init_population(min_depth, max_depth, population_size): # ramped half-and-half
    pop = []
    divisor = (max_depth - min_depth)*2
    for md in range(min_depth + 1, max_depth + 1):
        for i in range(int(population_size/divisor)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t)
        for i in range(int(population_size/divisor)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t)
    if len(pop) < population_size:
        for i in range(int((population_size - len(pop))/2)):
            md = randint(min_depth, max_depth)
            t = GPTree()
            t.random_tree(grow=True, max_depth=md)  # grow
            pop.append(t)
            t = GPTree()
            t.random_tree(grow=False, max_depth=md)  # grow
            pop.append(t)
    return pop


dataset = generate_dataset(rastrigin)
gp_results = dict()
gp_sim_data = defaultdict(list)
gp_data = defaultdict(list)
no_of_simulations = 1

# go evolution!
for sim in range(no_of_simulations):
    population = init_population(MIN_DEPTH, MAX_DEPTH, POP_SIZE)
    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
    for gen in range(GENERATIONS):
        nextgen_population = []
        for i in range(POP_SIZE):
            parent1 = selection(population, fitnesses, double=True)
            parent2 = selection(population, fitnesses, double=True)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population = nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            #print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses), 3), ", best_of_run:")
            #best_of_run.print_tree()
        gp_sim_data[sim].append(max(fitnesses))
        if best_of_run_f == 1: break
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(
        best_of_run_gen) + \
          " and has f=" + str(round(best_of_run_f, 3)))
    best_of_run.print_tree()

gp_data["Sphere_gen"] = [len(gp_sim_data[sim]) for sim in range(no_of_simulations)]
gp_data["Rastrigin_gen"] = [len(gp_sim_data[sim]) for sim in range(no_of_simulations)]
gp_data["Sphere_max"] = [max(gp_sim_data[sim]) for sim in range(no_of_simulations)]
gp_data["Rastrigin_max"] = [max(gp_sim_data[sim]) for sim in range(no_of_simulations)]
gp_data["Sphere_avg"] = np.average(np.array([max(gp_sim_data[sim]) for sim in range(no_of_simulations)]), axis=0)
gp_data["Rastrigin_avg"] = np.average(np.array([max(gp_sim_data[sim]) for sim in range(no_of_simulations)]), axis=0)
gp_data["Sphere_process"] = np.average(np.array([gp_sim_data[sim] for sim in range(no_of_simulations)]), axis=0)
gp_data["Rastrigin_process"] = np.average(np.array([gp_sim_data[sim] for sim in range(no_of_simulations)]), axis=0)

gp_df = pd.DataFrame(data=gp_data)
gp_df.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/gp_sphere.csv")




gp_results = defaultdict(dict)
gp_all_gen = dict()
gp_sim_data = defaultdict(list)
gp_data = defaultdict(list)
no_of_simulations = 30


def roulette_wheel_selection(population, dataset):
    # Computes the totallity of the population fitness
    population_fitness = sum([fitness(chromosome, dataset) for chromosome in population])

    # Computes for each chromosome the probability
    chromosome_probabilities = [fitness(chromosome, dataset) / population_fitness for chromosome in population]

    # Selects one chromosome based on the computed probabilities
    return np.random.choice(population, p = chromosome_probabilities)

class Particle_GP:

    def __init__(self, parent, dataset):
        self.position = parent
        self.new_position = None
        self.best_particle_pos = self.position
        self.dataset = dataset

        self.fitness = fitness(self.position, self.dataset)
        self.best_particle_fitness = self.fitness  # we couldd start with very large number here,
        # but the actual value is better in case we are lucky

    def setPos(self, pos):
        self.position = pos
        self.fitness = fitness(self.position, self.dataset)
        if self.fitness > self.best_particle_fitness:  # to update the personal best both
            # position (for velocity update) and
            # fitness (the new standard) are needed
            # global best is update on swarm leven
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updatePos(self, best_self_pos, best_swarm_pos):
        pbest = deepcopy(self.position)
        gbest = deepcopy(self.position)
        pbest.crossover(best_self_pos)
        pbest.mutation()
        gbest.crossover(best_swarm_pos)
        gbest.mutation()
        best_self_dif = pbest if fitness(pbest, self.dataset) > self.fitness else self.position
        best_swarm_dif = gbest if fitness(gbest, self.dataset) > self.fitness else self.position
        particle_population = [self.position, best_self_dif, best_swarm_dif]
        new_pos = roulette_wheel_selection(particle_population, self.dataset)
        self.new_position = new_pos
        return new_pos


class PSO_GP:

    def __init__(self, dataset, min_depth, max_depth, population_size, generations, sign_test=False, repeated=50,
                 p=0.00001):

        # Here we use values that are (somewhat) known to be good
        # There are no "best" parameters (No Free Lunch), so try using different ones
        # There are several papers online which discuss various different tunings of a1 and a2
        # for different types of problems
        self.population = init_population(min_depth, max_depth, population_size)
        self.fitnesses = [fitness(self.population[i], dataset) for i in range(population_size)]
        self.population_size = population_size
        self.p = p
        self.no_term = True
        self.S = 0
        self.sign_test = sign_test
        self.repeated = repeated
        self.dataset = dataset

        self.swarm = [Particle_GP(selection(self.population, self.fitnesses, double=True), dataset)
                      for i in range(population_size)]
        self.generations = generations
        self.best_of_run_f = 0
        self.best_of_run_gen = 0
        self.best_of_run = None
        print('init')

        # Initialising global best, you can wait until the end of the first time step
        # but creating a random initial best and fitness which is very high will mean you
        # do not have to write an if statement for the one off case
        self.best_swarm_pos = selection(self.population, self.fitnesses, double=True)
        self.best_swarm_fitness = -1e100
        self.best_swarm_fitness_record = []
        self.delta = []

    def run(self):
        for t in range(self.generations):
            nextgen_population = []
            for p in range(len(self.swarm)):

                particle = self.swarm[p]
                new_position = particle.updatePos(particle.best_particle_pos, self.best_swarm_pos)

                self.swarm[p].setPos(new_position)
                new_fitness = fitness(new_position, self.dataset)

                if new_fitness > self.best_swarm_fitness:  # to update the global best both
                    # position (for velocity update) and
                    # fitness (the new group norm) are needed
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position

                nextgen_population.append(new_position)

            self.delta = [self.best_swarm_fitness - i for i in self.best_swarm_fitness_record]
            self.best_swarm_fitness_record.append(self.best_swarm_fitness)

            self.S = 0
            for i in range(len(self.delta) - 1, 0, -1):
                if self.delta[i] == 0:
                    self.S += 1
                else:
                    break

            self.population = nextgen_population
            self.fitnesses = [fitness(self.population[i], self.dataset) for i in range(self.population_size)]

            if max(self.fitnesses) > self.best_of_run_f:
                self.best_of_run_f = max(self.fitnesses)
                self.best_of_run_gen = t
                self.best_of_run = deepcopy(self.population[self.fitnesses.index(max(self.fitnesses))])
                #print("________________________")
                print("gen:", t, ", best_of_run_f:", round(max(self.fitnesses), 3), ", best_of_run:")
                #self.best_of_run.print_tree()
            gp_all_gen[t] = self.best_swarm_fitness
            if self.best_of_run_f == 1: break


for i in range(no_of_simulations):
    y = PSO_GP(dataset, MIN_DEPTH, MAX_DEPTH, POP_SIZE, GENERATIONS)
    y.run()
    gp_results[i] = {"f": y.best_of_run_f, "gen": y.best_of_run_gen}
    gp_sim_data[i] = [gp_all_gen[j] for j in gp_all_gen.keys()]

gp_data["Sphere_gen"] = [gp_results[i]["gen"] for i in range(no_of_simulations)]
gp_data["Rastrigin_process"] = np.average(np.array([gp_sim_data[i] for i in range(no_of_simulations)]), axis=0)
gp_data["Rastrigin_avg"] = np.average(np.array([gp_results[i]["f"] for i in range(no_of_simulations)]))

gp_df = pd.DataFrame(data=gp_data)
gp_df.to_csv("/home/mich_qiu/PycharmProjects/nat-comp/gppso_rastrigin.csv")