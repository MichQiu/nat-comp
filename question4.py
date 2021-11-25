import numpy as np
import pandas as pd
import math
from random import random, randint, seed
from statistics import mean
from copy import deepcopy
from collections import defaultdict

def read_txt(textfile):
    data_list = []
    columns = []
    with open(textfile, "r") as f:
        count = 0
        for line in f:
            row = line.rstrip()
            data_list.append(row.split())
            columns.append(count)
            count += 1
        columns.pop(-1)
        data_list = np.array(data_list)
        df = pd.DataFrame(data_list, columns=columns)
        return df


POP_SIZE        = 60  # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
GENERATIONS     = 200  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 7    # size of tournament for tournament selection
XO_RATE         = 0.8  # crossover rate
PROB_MUTATION   = 0.2  # per-node mutation probability

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x,y): return x / y
def sin(x): return math.sin(x)
def cos(x): return math.cos(x)
def exp(x): return math.exp(x)
def abs(x): return np.abs(x)
def power_2(x): return x**2
def power_3(x): return x**3
def power_4(x): return x**4
def root_2(x): return x**(1/2)
def root_3(x): return x**(1/3)
def root_4(x): return x**(1/4)

FUNCTIONS = [add, sub, mul, div, sin, cos, exp, abs, power_2, power_3, power_4, root_2, root_3, root_4]
TERMINALS = ['x', 'y', 1, 2, 3, 4, 10, 100, np.pi]
unary_functions = [sin, cos, exp, abs, power_2, power_3, power_4, root_2, root_3, root_4]

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

    def compute_tree(self, x, y):
        if self.data in FUNCTIONS and self.data not in unary_functions:
            return self.data(self.left.compute_tree(x, y), self.right.compute_tree(x, y))
        elif self.data in unary_functions:
            if self.left is None:
                return self.data(self.right.compute_tree(x, y))
            else:
                return self.data(self.left.compute_tree(x, y))
        elif self.data == 'x':
            return x
        elif self.data == 'y':
            return y
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
            if self.best_of_run_f == 1: break

