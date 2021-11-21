# GP assignment

import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
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

                for i in range(len(new_veloctiy)):
                    if new_veloctiy[i] > self.vmax:
                        new_veloctiy[i] = self.vmax
                    elif new_veloctiy[i] < self.vmax:
                        new_veloctiy[i] = self.vmin


                new_position = particle.position + new_veloctiy

                for i in range(len(new_position)):
                    if new_position[i] > self.search_range:
                        new_position[i] = particle.position[i]
                        new_position[i] = -new_position[i]
                    elif new_position[i] < -self.search_range:
                        new_position[i] = particle.position[i]
                        new_position[i] = -new_position[i]

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
            '''
            if npsigntest(self.S, self.p) and self.no_term:
                print('Time:', t, 'Best Fit:', self.best_swarm_fitness, 'Best Pos:', self.best_swarm_pos,)
                self.no_term = False
                #raise SystemExit('Convergence: Termination condition satisfied')
            '''
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



for i in range(10):
    x = PSO(w=0.7, a1=2.02, a2=2.02, population_size=50, time_steps=3001, search_range=5.12, dim=7,
            func=rastrigin, a3=0.5, vl=0.5)
    x.run()


"""
# Contour plot: With the global minimum showed as "X" on the plot
x, y = np.array(np.meshgrid(np.linspace(-6, 6,9), np.linspace(-6,6,9)))
v = np.array([x,y])
z = rastrigin(v)
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]
plt.figure(figsize=(8,8))
plt.imshow(z, extent=[-6, 6, -6, 6], origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
plt.show()


def f(x, y):
    "Objective function"
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)

# Contour plot: With the global minimum showed as "X" on the plot
x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))
z = f(x, y)
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]
plt.figure(figsize=(8, 6))
plt.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
plt.show()
n_particles = 20
X = np.random.rand(2, n_particles) * 5
V = np.random.randn(2, n_particles) * 0.1
pbest = X
pbest_obj = f(X[0], X[1])
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([0,5])
ax.set_ylim([0,5])
plt.show()
"""