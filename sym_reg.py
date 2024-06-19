from random import random, uniform, sample, choice
from trees_utilities import *
from forest_utilities import *
import time
import csv
from math import exp, cos
import functions_dataset as fd
from copy import copy
from multiprocessing.pool import Pool
#import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import partial
import warnings
warnings.filterwarnings("ignore")

@dataclass
class Options:
    ecosystem_size: int = 40
    forest_size: int = 1000
    p_migration_best_sol: float = 0.05
    p_migration_other_forests: float = 0.05
    p_const_vs_var: float = 0.5
    constant_sd: float = 5
    initial_complexity: int = 3
    n_iterations: int = 100
    tournament_size: float = 12
    p_tournament: float = 0.9
    p_mutation_vs_cross: float = 0.99
    n_mutations: int = 300000
    complexity_limit: int = 30
    eps_tol: float = 10 ** -6
    annealing: float = 0.1
    constant_perturbation: float = 1
    minimum_perturbation: float = 0.1
    max_attempts: int = 10
    optim_max_iter: int = 5
    optim_method: str = 'Nelder-Mead'

class SRegression:
    def __init__(self,
                 data_X,
                 data_Y,
                 f_database = {},
                 max_variables = None,
                 **kwargs
                 ):
        '''
        Initializes and runs the Symbolic Regression algorithm.

            Parameters:
                data_X (list):                     Matrix or list of observed independent variables' values.
                data_y (list):                     Vector of observed dependent variable's values. Namely, Y = f(X).
                f_database (dict):                 Dict or set whose keys are the names of the functions that we want to consider on our trees.
                                                    If a dict is introduced, the values will be the weight considered when randomly generating
                                                    its corresponding key, else it consideres the standard weights predefined in the complete_functions_database.
                                                    Default uses the whole standard database and weights.
                max_variables (int):               Maxium number of variables that a Tree may take.
                ecosystem_size (int):              Number of different forests that will be developed during the algorithm. Default is 40.
                forest_size (int):                 The number of trees that each forest will have. Default is 1000.
                p_migration_best_sol (float):      Probability of migrating the best solution to a given forest. Default is 0.05.
                p_migration_other_forests (float): Probability of migrating one of the best trees from a random forest to the given one. Default is 0.05.
                p_const_vs_var (float):            The probability of selecting a constant vs a variable when generating trees. Default is 0.5
                constant_sd (float):               Standard deviation for the generation of constants. Default is 10 
                initial_complexity (int):          The number of nodes that trees will initially take when randomly generated. Default is 3
                n_iterations (int):                Maximum number of times the algorithm will iterate. Default is 100
                tournament_size (int):             Number of trees that will compete on a tournament. Default is 12
                p_tournament (float):              Probability of choosing the tree with least error on a tournament. Default is 0.9
                p_mutation_vs_cross (float):       Probability of implementing a mutation vs a crossover. Default is 0.99
                n_mutations (int):                 Number of mutations that will be performed on each iteration of the algorithm. Default is 300000
                complexity_limit (int):            Maximum number of nodes a tree may take. Default is 30
                eps_tol (float):                   Tolerance needed to stop the iterations. That is to say, if the best solution error is less than
                                                    eps_tol, the regression will be terminated. Default is 10 ** -6
                annealing (float):                 Annealing factor. Default is 0.1
                constant_perturbation (float):     Constant perturbation scale for constant mutation. Default is 1
                minimum_perturbation (float):      Minimum constant perturbation for constant mutation. Default is 0.1
                max_attempts (int):                Maximum number of attempts to be performed when doing a crossover or mutation. Default is 10   
                optim_max_iter (int):              Maximum number of iterations to be performed when optimizing an expression. Default is 5
                optim_method (str):                Method to be performed when optimizing an expression. Default is 'Nelder-Mead'
        '''
        # We initialize the options dataclass.
        options = Options()
        # We add some parameters that are not default and are mandatory to specify.
        options.f_database = f_database
        options.data_X = data_X
        options.data_Y = data_Y
        options.max_variables = max_variables
        # Now we assign the optional parameters.
        for name, value in kwargs.items():
            setattr(options, name, value)
        # We call for the initialization of the global variables on the Trees' methods and functions.
        init_problem(options)
        # ## ALGORITHM ##
        # We start the time counter.
        self.run_time = time.time()
        # We generate the ecosystem (a group of n forests) with a Pool.
        with Pool() as pool:
            ecosystem = pool.map(Forest, [options for _ in range(options.ecosystem_size)])
        # We initialize the global best solutions dictionary.
        self.best_solutions = {key: 0 for key in range(2, options.complexity_limit + 1)}
        print('{: <5} {: ^100} {: <20} {: <10} {: <10}'.format('It.', 'Best solution', 'Best solution MSE', 'Age (s)', 'Run time'))
        # ## ALGORITHM LOOP ##
        for i in range(options.n_iterations):
            # ## EVOLVING ##
            # Evolving with Pool.
            with Pool() as pool:
                 ecosystem = pool.map(partial(Forest.evolve, options = options), ecosystem)
            # ## UPDATING BEST SOLUTIONS
            for forest in ecosystem:
                # We update the dictionary of best expressions.
                for complexity, tree in forest.best_solutions.items():
                    best_solution_at_complexity = self.best_solutions[complexity]
                    if tree != 0 and (best_solution_at_complexity == 0 or best_solution_at_complexity.error > tree.error):
                        self.best_solutions[complexity] = tree
            # ## MIGRATION MECHANISM ##
            # For each Forest.
            for forest_index, forest in enumerate(ecosystem):
                # For each Tree on said Forest, the Tree will migrate with total probability options.p_migration_best_sol + options.migration_other_forests.
                for j, tree in enumerate(forest.trees):
                    # With probability options.p_migration_best_sol, we select one of the functions on the overall best_solutions dictionary and
                    # incorporate it to the Forest.
                    if R := random() < options.p_migration_best_sol:
                        # We delete the given Tree.
                        del forest.trees[j]
                        # And insort a random Tree selected from the overall best solutions.
                        insort(forest.trees, b_sol := deepcopy(choice([value for value in self.best_solutions.values() if value != 0])), key = lambda x: - x.birth)
                        # We update the complexity dictionary of the Forest.
                        forest.complexity_dictionary[tree.n_nodes] -= 1
                        forest.complexity_dictionary[b_sol.n_nodes] += 1
                    # With probability options.p_migration_other_forest, we select one of the functions on any of the other Forest's best_solutions
                    # dictionary and incorporate it to the current one.
                    elif R < options.p_migration_other_forests + options.p_migration_best_sol:
                        # We delete the given Tree.
                        del forest.trees[j]
                        other_forest = ecosystem[choice([i for i in range(options.ecosystem_size) if i != forest_index])]
                        insort(forest.trees, b_sol := deepcopy(choice([value for value in other_forest.best_solutions.values() if value != 0])), key = lambda x: - x.birth)
                        # We update the complexity dictionary of the Forest
                        forest.complexity_dictionary[tree.n_nodes] -= 1
                        forest.complexity_dictionary[b_sol.n_nodes] += 1       
            # We calculate the best overall solution:
            self.best_solution = min([value for value in self.best_solutions.values() if value != 0], key = lambda x: x.error)
            if self.best_solution.error / len(data_X) < options.eps_tol:
                break
            print('{: <5} {: ^100.95} {: <20.5e} {: <10.2f} {: <10.2f}'.format(i + 1, truncate(str(self.best_solution), 95), self.best_solution.error / len(data_X), time.time() - self.best_solution.birth / 10 ** 7, time.time() - self.run_time))
            #self.display_solutions()
        print('{: <5} {: ^100.95} {: <20.5e} {: <10.2f} {: <10.2f}'.format('Sol.:', truncate(str(self.best_solution), 95), self.best_solution.error / len(data_X), time.time() - self.best_solution.birth / 10 ** 7, time.time() - self.run_time))
        # ## EXECUTION DATA STORING:
        # We store the number of iterations that the algorithm needed:
        self.n_it = i
        # We store the total time of execution:
        self.run_time = time.time() - self.run_time
        self.display_solutions(options)

    def display_solutions(self, options):
        '''
        Method that prints the best solutions obtained ordered by complexity.
        It shows the expression, its complexity and the error.
        '''
        # We sort the solutions by complexity:
        solutions = sorted(self.best_solutions.items())
        # We print them:
        print('{: <5} {: ^150} {: <20} {: <10}'.format('Comp.', 'Solution', 'Solution error', 'Age'))
        for sol in solutions:
            if sol[1] == self.best_solution:
                print('{: <5} {: >10} {: ^90.85} {: <20.5e} {: <10.2f}'.format(sol[0], '>>>>>>>', truncate(str(sol[1]), 85), sol[1].error / len(options.data_X), sol[1].birth / 10 ** 7))
            elif sol[1] != 0:
                print('{: <5} {: ^100.95} {: <20.5e} {: <10.2f}'.format(sol[0], truncate(str(sol[1]), 95), sol[1].error / len(options.data_X), sol[1].birth / 10 ** 7))

def truncate(string, width):
    '''
    Function used to format long equations.
    '''
    if len(string) > width:
        string = string[:width-3] + '...'
    return string

if __name__ == '__main__':
    # histogrammer = dict()
    # for objective_function, obj_fun_data in fd.livermore.items():
    #     obj_func = obj_fun_data[0]
    #     print(objective_function)
    #     data_X = [[uniform(obj_fun_data[2], obj_fun_data[3])
    #                         for _ in range(obj_fun_data[1])] for _ in range(obj_fun_data[4])]
    #     data_Y = [obj_func(*x) for x in data_X]
    #     reg = SRegression(data_X, data_Y, f_database = {'+', '-', '*', '^', '―', '/', 'sqrt', 'exp', 'log', 'cbrt', 'exp2', 'log2', 'sinh', 'cosh', 'asinh', 'gamma', 'sin', 'cos', 'asin', 'acos'},
    #                         n_iterations = 15, n_mutations = 2500, complexity_limit = 50,
    #                         ecosystem_size = 20, p_mutation_vs_cross = 0.99)
    #     #histogrammer[objective_function] = reg.best_solution.mutations
    #     #histogrammer[objective_function]['it-error'] = (reg.n_it, reg.run_time, reg.best_solution.error)
    #     #print(reg.best_solution.mutations)
    # for key, value in histogrammer.items():
    #     print(key, value)

    with open('beam-data-T-CFRP-U.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data_X = []
        data_Y = []
        for row in reader:
            if row != []:
                data_X.append(row[0:-2])
                data_Y.append(row[-1])
        data_X = data_X[1:]
        data_Y = data_Y[1:]
        data_Y = [float(dat) for dat in data_Y]
        indexes = {0, 1, 2, 4, 11}
        indexes_extended = {0, 1, 2, 4, 7, 9, 11}
        indexes_extended2 = {0, 1, 2, 3, 4, 7, 9, 11} # This has missing values
        data_X2 = []
        for k, data in enumerate(data_X):
            data_X2.append([float(dat) for i, dat in enumerate(data) if i in indexes])
        
        vigas = SRegression(data_X2, data_Y, f_database = {'+', '-', '*', '^', '―', '/', 'gamma', 'sigm', 'abs', 'ceil'},
                            complexity_limit = 40, n_mutations = 10000, n_iterations = 50, forest_size = 1000, ecosystem_size = 40, max_variables = 3)
        print(vigas.best_solution)
        print(vigas.best_solution.evaluate(data_X2))
