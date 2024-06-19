from trees_utilities import *
from collections import Counter
from random import sample
from math import inf, exp
from bisect import insort

class Forest:
    def __init__(self, options):
        '''
        Method that generates a forest given its size. Each one of the trees has fixed initial complexity.

            Parameters:
                options (Options): Custom parameters of the algorithm execution.  
        '''
        # Container variable.
        trees = []
        # We generate exactly forest_size trees.
        while len(trees) < options.forest_size:
            try:
                # We generate a random tree.
                tree = Tree(options, n_nodes = options.initial_complexity)
                # We also calculate its error.
                tree.error_calc(options)
                # If the tree is well defined, we add it to the forest.
                if isinstance(tree.error, float) and isfinite(tree.error):
                    trees += [tree]
            except:
                pass        
        # We sort the forest by increasing age.
        trees.sort(key = lambda x: - x.birth)
        # We save the generated trees as an attribute of the forest.
        self.trees = trees
        # We initialize the complexity dictionary.
        self.complexity_dictionary = {key: 0 for key in range(2, options.complexity_limit + 1)}
        self.complexity_dictionary[options.initial_complexity] = options.forest_size
        # We create an empty dictionary to store the best solutions of the forest.
        self.best_solutions = {key: 0 for key in range(2, options.complexity_limit + 1)}
    
    def tournament(self, options):
        '''
        Algorithm that selects a subset of tournament_size elements.
        Then, it chooses its fittest individual with probability p_tour,
        its second fittest individual with probability p_tour * (1-p_tour), etc.
        The fittest individual is not the Tree with least error.
        We take into account both its accuracy and complexity (we want to evolve all complexities equally).

            Parameters:
                options (Options): Custom parameters of the algorithm execution.     
            Returns:
                fittest (Tree):    Selected tree on the performed tournament.
        '''
        # We selected a random subgroup of tournament_size Trees within the given forest.
        tournament_group = sample(self.trees, options.tournament_size)
        # Tournament loop to select the fittest within the selected Trees:
        while len(tournament_group) > 1:
            # We initialize the lowest error.
            best_error = inf
            # We initialize the index counter.
            j = 0
            # We will select the current fittest tree of the tournament group.
            for i, tree in enumerate(tournament_group):
                # We want to punish the complexities that are dominating the given forest.
                # We should have around forest_size/complexity_limit Trees at each complexity.
                # So we will try to select trees from complexities that have less Trees than the expected forest_size/complexity_limit.
                # That's why we apply a function (adapted from PySR) that takes this into account.
                current_error = tree.error * exp(20 * self.complexity_dictionary.get(tree.n_nodes, 0) / options.forest_size)
                # (This one is another option of punishing, but made by myself):
                # current_error = tree.error * (1 - exp(-(2.079 * self.complexity_dictionary.get(tree.n_nodes, 0)) / (3 * (options.forest_size / options.complexity_limit))))
                # Now, we want to select the tree with less rescaled error.
                if current_error < best_error:
                    best_error = current_error
                    j = i
                    fittest = tree
            # We return the current fittest tree with probability p_tournament.
            if random() < options.p_tournament:
                break
            # With probability 1 - p_tournament we don't, and so we delete it from the group.
            del tournament_group[j]
        # If there is only one function left in the tournament group, we have to return it.
        if len(tournament_group) == 1:
            return tournament_group[0]
        # We return the tournament winner.
        return fittest
    
    def evolve(self, options):
        '''
        Function that evolves a given forest.
        That is to say, it performs up to n_mutations mutations or crossovers on the forest's Trees selected by tournament.
        
            Parameters:
                options (Options): Custom parameters of the algorithm execution.
        '''
        # We perform n_mutations mutations or crossovers:
        for i in range(options.n_mutations):
            # ## MUTATIONS
            # With probability p_mutation_vs_cross we mutate a tree:
            if random() < options.p_mutation_vs_cross:
                # Annealing temperature parameter:
                temperature = 1 - i / options.n_mutations
                # We select the tree to be mutated by a tournament:
                tree = self.tournament(options)
                #complexity_before_mutation = tree.n_nodes
                # We mutate it:
                tree, succesful_mutation = mutate(tree, temperature, self.complexity_dictionary, options)
                if succesful_mutation:
                    # We update the complexity dictionary:
                    self.complexity_dictionary[self.trees[-1].n_nodes] -= 1
                    # We eliminate the last tree of the forest (the oldest one):
                    del self.trees[-1]
                    # We insert the new tree without altering the order of the forest:
                    insort(self.trees, tree, key = lambda x: - x.birth)
                    self.complexity_dictionary[tree.n_nodes] += 1
            # ## CROSSOVERS
            # With probability 1 - p_mutation_vs_cross we procceed to make a crossover
            else:
                # We select the trees by a tournament and crossover them:
                tree_A = self.tournament(options)
                tree_B = self.tournament(options)
                #complexity_before_mutation_A = tree_A.n_nodes
                #complexity_before_mutation_B = tree_B.n_nodes
                tree_A, tree_B, crossover_succesful = crossover(tree_A, tree_B, options)
                if crossover_succesful:
                    self.complexity_dictionary[self.trees[-1].n_nodes] -= 1
                    self.complexity_dictionary[self.trees[-2].n_nodes] -= 1
                    del self.trees[-2:]
                    insort(self.trees, tree_A, key = lambda x: - x.birth)
                    insort(self.trees, tree_B, key = lambda x: - x.birth)
                    self.complexity_dictionary[tree_A.n_nodes] += 1
                    self.complexity_dictionary[tree_B.n_nodes] += 1
        # Simplification and optimization of all the expressions on the forest:
        for i, tree in enumerate(self.trees):
            # ## SIMPLIFICATION
            tree = simplify(tree, options)
            tree.constant_finder([], [], False)
            # ## OPTIMIZATION
            # If the tree has constants, we optimize it.
            if tree.constants != []:
               tree = tree.optimize(options)
            #tree.mutations['gen'] += 1
            self.trees[i] = tree
            # We update the dictionary of best expressions of the forest.
            best_solution_at_complexity = self.best_solutions[tree.n_nodes]
            # If we don't have a stored best solution, we add one. 
            if best_solution_at_complexity == 0:
                self.best_solutions[tree.n_nodes] = tree
            # Else, we have to check if we've upgraded the current error.
            elif best_solution_at_complexity.error > tree.error:
                self.best_solutions[tree.n_nodes] = tree
        return self