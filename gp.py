# GP assignment

import numpy as np
import math
from random import random, randint, seed
from statistics import mean
from copy import deepcopy

# sphere function
def sphere(x):
    return sum(np.square(x))

# Rastrigin function
def rastrigin(x):
    return 10*len(x) + sum(np.square(x)-10*np.cos((2*np.pi*x)))

# non-parametric sign test
def npsigntest(S, p=0.001):
    ## S (int): the number of zero difference fitnesses counting from the last fitness in sequence
    ## p (float): the error bound probability
    if S > 1:
        cumulative = []
        for i in range(S - 1, 1, -1):
            prob = ((math.factorial(S)) / ((math.factorial(i)) * (math.factorial(S - i)))) * (1 / 2) ** S
            cumulative.append(prob)
            if sum(cumulative) <= p:
                return True
        return False
    else:
        return False


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

    def __init__(self, w, a1, a2, population_size, time_steps, search_range, dim, func, p=0.00001, a3=None, vl=None):

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
        self.vl = vl
        self.search_range = search_range

        self.vmax = vl*search_range
        self.vmin = vl*(-search_range)
        self.swarm = [Particle(func, dim, -search_range, search_range) for i in range(population_size)]
        self.time_steps = time_steps
        print('init')

        # Initialising global best, you can wait until the end of the first time step
        # but creating a random initial best and fitness which is very high will mean you
        # do not have to write an if statement for the one off case
        self.best_swarm_pos = np.random.uniform(low=-500, high=500, size=dim)
        self.best_swarm_fitness = 1e100
        self.best_swarm_fitness_record = []
        self.delta = []

    def run(self):
        for t in range(self.time_steps):
            for p in range(len(self.swarm)):
                particle = self.swarm[p]

                repulse_terms = []
                r3 = np.random.uniform(low=0, high=1, size=self.dim)
                all_other = deepcopy(self.swarm)
                del all_other[p]
                for k in all_other:
                    neighbour = np.subtract(k.position, particle.position)
                    x = (np.multiply(r3, neighbour))/((np.linalg.norm(neighbour))**2)
                    repulse_terms.append(x)
                Z = -sum(repulse_terms)

                new_veloctiy =  particle.updateVel(self.w, self.a1, self.a2,
                                                                      particle.best_particle_pos, self.best_swarm_pos,
                                                                      self.a3, Z, repulse=False)

                '''
                for i in range(len(new_veloctiy)):
                    if new_veloctiy[i] > self.vmax:
                        new_veloctiy[i] = self.vmax
                    elif new_veloctiy[i] < self.vmax:
                        new_veloctiy[i] = self.vmin
                '''
                new_position = particle.position + new_veloctiy
                '''
                for i in range(len(new_position)):
                    if new_position[i] > self.search_range:
                        new_position[i] = particle.position[i]
                        new_position[i] = -new_position[i]
                    elif new_position[i] < -self.search_range:
                        new_position[i] = particle.position[i]
                        new_position[i] = -new_position[i]
                '''
                if new_position @ new_position > 1.0e+18:  # The search will be terminated if the distance
                    # of any particle from center is too large
                    print('Time:', t, 'Best Pos:', self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                    raise SystemExit('Most likely divergent: Decrease parameter values')

                self.swarm[p].setPos(new_position)
                new_fitness = self.func(new_position)

                if new_fitness < self.best_swarm_fitness:  # to update the global best both
                    # position (for velocity update) and
                    # fitness (the new group norm) are needed
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position

            self.delta = [self.best_swarm_fitness - i for i in self.best_swarm_fitness_record]
            self.best_swarm_fitness_record.append(self.best_swarm_fitness)

            self.S = 0
            for i in range(len(self.delta) - 1, 0, -1):
                if self.delta[i] == 0:
                    self.S += 1
                else:
                    break

            if npsigntest(self.S, self.p) and self.no_term:
                print('Time:', t, 'Best Fit:', self.best_swarm_fitness, 'Best Pos:', self.best_swarm_pos,)
                self.no_term = False
                #raise SystemExit('Convergence: Termination condition satisfied')

            if self.best_swarm_fitness == 0:
                print('Time:', t, 'Best Fit:', self.best_swarm_fitness, 'Best Pos:', self.best_swarm_pos,)
                break

            if t % 100 == 0:  # we print only two components even it search space is high-dimensional
                print("Time: %6d,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (
                t, self.best_swarm_fitness, self.best_swarm_pos[0], self.best_swarm_pos[1]), end=" ")
                if self.dim > 2:
                    print('...')
                else:
                    print('')
'''
for i in range(10):
    x = PSO(w=0.7, a1=2, a2=2, population_size=30, time_steps=5001, search_range=5.12, dim=30,
            func=rastrigin, a3=2, vl=0.5)
    x.run()
'''

POP_SIZE        = 60  # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 8    # maximal initial random tree depth
GENERATIONS     = 500  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 7    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate
PROB_MUTATION   = 0.2  # per-node mutation probability

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def cos(x): return math.cos(x)

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

def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])

def init_population(): # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE/12)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t)
        for i in range(int(POP_SIZE/12)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t)
    return pop


dataset = generate_dataset(rastrigin)
population = init_population()
best_of_run = None
best_of_run_f = 0
best_of_run_gen = 0
fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]

# go evolution!
for gen in range(GENERATIONS):
    nextgen_population = []
    for i in range(POP_SIZE):
        parent1 = selection(population, fitnesses)
        parent2 = selection(population, fitnesses)
        parent1.crossover(parent2)
        parent1.mutation()
        nextgen_population.append(parent1)
    population = nextgen_population
    fitnesses = [fitness(population[i], dataset) for i in range(POP_SIZE)]
    if max(fitnesses) > best_of_run_f:
        best_of_run_f = max(fitnesses)
        best_of_run_gen = gen
        best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])
        print("________________________")
        print("gen:", gen, ", best_of_run_f:", round(max(fitnesses), 3), ", best_of_run:")
        best_of_run.print_tree()
    if best_of_run_f == 1: break

print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(
    best_of_run_gen) + \
      " and has f=" + str(round(best_of_run_f, 3)))
best_of_run.print_tree()