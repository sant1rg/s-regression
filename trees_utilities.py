import operator as op
import numpy as np
import math
from random import randint, choices, random
from numpy import random as rd
from numpy import isfinite
from copy import deepcopy, copy
import time
from scipy import optimize
from sys import exit

# ## DATABASE OF OPERATORS AND FUNCTIONS ##
# We define some hand-made functions that will be included on the database:
def neg(x):
    return -x
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def ceil(x):
    return float(math.ceil(x))
def floor(x):
    return float(math.floor(x))
# Database:
complete_functions_database = {
# Meaning of the elements of the tuple:
#       Position 0: Function in python.
#       Position 1: Domain/mutation group. Functions with the same domain
#                   will be interchangeable when mutating.
#       Position 2: Arity of the function.
#       Position 3: Function's weight when randomly generating a tree.
#       Position 4: Boolean indicating in which order the argument
#                   name and its arguments are written when converting to string.
#       Position 5: Inverse of the given function (if it exists).
# Groups of mutation:
#       0: Operators. Domain = R x R
#       1: Domain = R
#       2: Domain = (0, inf)
#       3: Domain = (-pi/2, pi/2)
#       4: Domain = (-1, 1)
#       5: Domain = (1, inf)
    '+':     (op.add,     0, 2, 5, True),
    '-':     (op.sub,     0, 2, 5, True),
    '*':     (op.mul,     0, 2, 5, True),
    '/':     (op.truediv, 0, 2, 5, True),
    '^':     (math.pow,   0, 2, 5, True),
    '―':     (op.neg,     1, 1, 5, False),

    'sigm':  (sigmoid, 1, 1, 3, False),
    'sqrt':  (math.sqrt,  2, 1, 3, False),
    'exp':   (math.exp,   1, 1, 3, False, 'log'),
    'log':   (math.log,   2, 1, 5, False, 'exp'),
    'sin':   (math.sin,   1, 1, 3, False, 'asin'),
    'cos':   (math.cos,   1, 1, 3, False, 'acos'),
    'tan':   (math.tan,   3, 1, 1, False, 'atan'),
    'ceil':  (ceil,  1, 1, 1, False, 'floor'),
    'abs':   (math.fabs,  1, 1, 1, False, 'abs'),
    'floor': (floor, 1, 1, 1, False, 'ceil'),
    'cbrt':  (math.cbrt,  1, 1, 1, False),
    'exp2':  (math.exp2,  1, 1, 1, False, 'log2'),
    'log2':  (math.log2,  2, 1, 1, False, 'exp2'),
    'log10': (math.log10, 2, 1, 1, False),
    'acos':  (math.acos,  4, 1, 1, False, 'cos'),
    'asin':  (math.asin,  4, 1, 1, False, 'sin'),
    'atan':  (math.atan,  1, 1, 1, False, 'tan'),
    'cosh':  (math.cosh,  1, 1, 1, False, 'acosh'),
    'sinh':  (math.sinh,  1, 1, 1, False, 'asinh'),
    'tanh':  (math.tanh,  1, 1, 1, False, 'atanh'),
    'asinh': (math.asinh, 1, 1, 1, False, 'sinh'),
    'acosh': (math.asinh, 5, 1, 1, False, 'cosh'),
    'atanh': (math.asinh, 4, 1, 1, False, 'tanh'),
    'gamma': (math.gamma, 2, 1, 3, False)
}
n_mut_groups = 6
# ## CLASSES ##
class Tree:
    def __init__(self, options, name = None, parent = None, children = None, n_nodes = 3):
        '''
        A tree is an object that consists of a function or operator node with its children.
        Variables and constants are not considered as tree objects, as they are always terminal
        nodes and can be declared as variable strings 'xi' or floats/integers.

            Parameters:
                name (str):           The name of the function or operator the tree carries. Default is None
                parent (Tree):        The tree's parent if it is included as a children of said parent. Default is None
                children (list):      A list containing the tree's child nodes. Default is None
                random (bool):        Set to True if we want to randomly generate the tree. Default is True
                n_nodes (int):        In case of random generation, the maximum depth the tree will take. Default is 3
        '''
        # We set its birth time:
        self.birth = time.time() * 10 ** 7
        if name is None:
            # If random is set to true, we call a generator function.
            self.generate(options, n_nodes, parent)
        else:
            # We assign its name, parent, and children list.
            self.name = name
            self.children = children
            self.parent = parent
            # We load the data corresponding to its name from the function and operators database:
            func_data = options.functions[self.name]
            # We load the Python function corresponding to its name:
            self.func = func_data[0]
            # We load a boolean that specifies how to write the function when converting to string:
            self.writing_order = func_data[4]
            # Number of inputs its corresponding Python function takes:
            self.arity = func_data[2]
            self.n_nodes = n_nodes
        # We search for the Tree's constants and store them.    
        self.constant_finder([], [], False)
        # Information of the mutations that were done to the function.
        #self.mutations = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0], 8: [0, 0], 9: [0, 0], 'optimized': [0, 0], 'gen': 0, 'crossover': [0, 0]}

    def __str__(self):
        '''
        Recursive method that returns a string with the tree's expression.
        It adds parentheses to every operation even when the order is clear to simplify the code.

            Returns:
                A string portraying the tree's mathematical expression
        '''
        # We convert to string all the node's children:
        children_strings = ['x' + str(child + 1) if isinstance(child, int)
                             else str(child) for child in self.children]

        # We have to discern between funcions and operators as its written syntax is different:
        # a1 * ... * an for an operator,
        # f(a1, ..., an) for a function.
        # writing_order = True means that we are going to write the tree as an operator:
        if self.writing_order:
            return '(' + (' ' + self.name + ' ').join(children_strings) + ')'
        # Else, we write the tree as a function:
        else:
            return self.name + '(' + ', '.join(children_strings) + ')'

    def evaluate(self, data_list):
        '''
        Recursive method that evaluates a tree in a list of vectors (input points).

            Parameters:
                data_list (list):  Points of data where we are going to evaluate the expression.                    
            Returns:
                list:              The list of values of the expression in each of the data points
        '''

        # We evaluate each of the node's children in the list of input points:
        evals = [child.evaluate(data_list) if isinstance(child, Tree)
                                      else [point[child] for point in data_list] if isinstance(child, int)
                                      else [child] * len(data_list)
                     for child in self.children] 
        #evals has the form [[list of evaluation for child1], 
        #					[list of evaluation for child2].....]   
        #map(self.func,*evals) evaluates self.func with a element of each child's list as parameters     
        # We return the list of evaluations of the node's function in the list of points:
        return list(map(self.func, *evals))
        # The asterisk converts the list to each of the individual parameters the function may take.
        
    def generate(self, options, n_nodes, parent = None):
        '''
        Recursive method that generates a random tree.

            Parameters:
                options (Options): Custom parameters of the algorithm execution.  
                n_nodes (int):     The total number of nodes that the generated tree will have
                parent (Tree):     The parent of the tree that we are randomly generating. Default is None
        '''
        # We start by sampling randomly an operator or a function.
        # We have to make sure that the chosen function does not require more nodes than n_nodes.
        # For example if n_nodes = 2, we cannot select an operator with arity 2 (because its arity + the operator itself
        # would make up to more than 3 nodes).
        while True:
            # We randomly select a function:
            selected_fun = choices(options.func_names, options.func_weights)[0]
            # We make sure that we won't exceed n_nodes:
            if (options.functions[selected_fun][2] + 1) <= n_nodes:
                self.name = selected_fun
                break
        # After selecting the proper function, we set the node's attributes:
        func_data = options.functions[self.name]
        self.parent = parent
        self.func = func_data[0]
        self.writing_order = func_data[4]
        self.arity = func_data[2]
        self.n_nodes = n_nodes

        # We have just generated one node. We need to fill up the remaining n_nodes - 1:
        descendants = n_nodes - 1
        # If the arity of the node is 1, we just generate one child:
        if self.arity == 1:
            self.children = [
                             # We generate a tree if we have to fill up more than one node:
                             Tree(parent = self, n_nodes = descendants, options = options) if descendants > 1
                             # Else we generate a function or variable:
                             else round(rd.normal(scale = options.constant_sd), randint(0, 15)) if random() < options.p_const_vs_var
                             else randint(0, options.n_var - 1)]
        # Else, if the arity is greater than one, we will allocate the remaining nodes on its different input positions:
        else:
            # This variable is to make evaluations faster:
            slots = self.arity - 1
            # Now, on each of the input positions of the node, we need to assign at least one node:
            descendants_partition = self.arity * [1]
            # We then procceed to allocate the remaining nodes uniformly:
            for _ in range(descendants - self.arity):
                descendants_partition[randint(0, slots)] += 1
            # At last, we randomly generate subtrees with the number of nodes given by the recent allocation:
            self.children = []
            for i in range(self.arity):
                self.children += [
                                  # We generate a tree if we have to fill up more than one node on the i-th position of the input:
                                  Tree(parent = self, n_nodes = descendants_partition[i], options = options) if descendants_partition[i] > 1
                                  # Else we generate a function or variable:
                                  else round(rd.normal(scale = options.constant_sd), randint(0, 15)) if random() < options.p_const_vs_var
                                  else randint(0, options.n_var - 1)]


    def node_chooser(self, k, parent = None, i = 0):
        '''
        Method that search the descendant node in position k when counting in depth.

            Parameters:
                k (int):            The position of the node we are searching for
                parent (Tree):      Variable to be modified inside recursion. It indicates the parent of the selected node. Default is None
                i (int):            Variable to be modified inside recursion. It points to the position of the selected node inside its
                                     parent's children list. Default is None
            Returns:
                parent, i (tuple): The first element of the tuple is the desired node's parent and the second element its position inside its
                                    parent's node children list. Namely, the desired node is parent[i]. If we search for the root node the
                                    result parent will be None and the desired node will be self
        '''
        # If k is equal to self.n_nodes, it means that the node itself is the selected one.
        if k == self.n_nodes: 
            return (parent, i)
        # Else, we will search between its descendants.
        # We will start with its first child. We count how many descendant it has, if it is less than k,
        # then the desired node is not between that child's descendants. We repeat with the second child and so on...
        else:
            # This variable keeps track of the nodes we have searched:
            desc_count = 0
            for j, child in enumerate(self.children): 
                # For each child node, we count the number of descendants it has (including itself).
                # If it the child is a Tree, we use the attribute n_nodes:
                if isinstance(child, Tree):
                    desc_count += child.n_nodes
                # Else, it only has one descendant (itself):
                else: desc_count += 1
                # We finish when we have counted more nodes than k:
                if desc_count >= k:
                    # If the child is a tree, we have to refine the search (the desired node is between its children):
                    if isinstance(child, Tree):
                        # We continue the search updating the position we are searching for,
                        # the parent node, and the position inside that parent node.
                        return child.node_chooser(k - (desc_count - child.n_nodes), self, j)
                    # If it is a variable or constant, we finish:
                    return (self, j)

    def error_calc(self, options):
        '''
        Method that stores the error of an expression given a data matrix.
        Right now it calculates the MSE, but it may be modified.
        
            Parameters:
                options (Options): Custom parameters of the algorithm execution.  
        '''
        self.error = sum(np.subtract(self.evaluate(options.data_X), options.data_Y) ** 2)
             
    def constant_finder(self, constants = [], variables = [], constantify = False):
        '''
        Method that searches and stores on the attribute constants the constants of the tree.
        It also modifies the constants to negative integers in order to be evaluated later.

            Parameters:
                constants (list):   List that contains the constants of the tree. To be modified inside recursion
                constantify (bool): True if the Tree is going to be constantified (in order to change constants later).
        '''
        # We search between the tree's children:
        for i, child in enumerate(self.children):
            # If the given child is a tree, we have to keep searching for constants:
            if isinstance(child, Tree):
                child.constant_finder(constants, variables, constantify)
            # Else if the given child is a constant:
            elif isinstance(child, float):
                # We store the constant on the constants list:
                constants.append(child)
                if constantify:
                    # We change the constant to a negative integer:
                    self.children[i] = - len(constants)
            elif isinstance(child, int):
                if child not in variables:
                    variables.append(child)
        # When the search is finished, we store the constants on the constants attribute:
        self.constants = constants
        self.variables = variables

    def evaluate_with_constants(self, data_list, constants):
        '''
        Recursive method that evaluates a constantified tree given the data to evaluate and the values the constants of
        the tree are going to take.

            Parameters:
                data_list (list): Points of data where we are going to evaluate the expression.      
                constants (list): A vector containing the values that the constants c1, c2, ..., cm will take
            Returns:
                float:            The value of the evaluated expression
        '''
        # We evaluate each of the node's children in the list of input points:
        evals = [child.evaluate_with_constants(data_list, constants) if isinstance(child, Tree)
                                      else [point[child] for point in data_list] if child >= 0
                                      else [constants[- (child + 1)]] * len(data_list)
                     for child in self.children] 
        #evals has the form [[list of evaluation for child1], 
        #					[list of evaluation for child2].....]   
        #map(self.func,*evals) evaluates self.func with a element of each child's list as parameters     
        # We return the list of evaluations of the node's function in the list of points:
        return list(map(self.func, *evals))
        # The asterisk converts the list to each of the individual parameters the function may take.
    
    def error_calc_with_constants(self, options, constants):
        '''
        Method that stores the error of a constantified expression given a data matrix and the values of the constants.
        Right now it calculates the MSE, but it may be modified.
        
            Parameters:
                data (list):         Data matrix of given independent variables
                constants (list):    A vector containing the values that the constants c1, c2, ..., cm will take
                observations (list): Observations of the dependent variable
            Returns:
                float:               The calculated error 
        '''
        return sum(np.subtract(self.evaluate_with_constants(options.data_X, constants), options.data_Y) ** 2)  

    def undo_constants(self, constants):
        '''
        Method that, given a constantified tree, replaces the negative integer values associated with constants
        with the given constants list.

            Parameters:
                constants (list): The vector of constants that will be replaced in the tree's negative integers
        '''
        # We search for the negative integers in the tree's nodes:
        for i, child in enumerate(self.children):
            # If the child is a Tree, we have to keep searching:
            if isinstance(child, Tree):
                child.undo_constants(constants)
            # Else, all the nodes are integers. If it is negative, it is a constant that will be replaced
            # by its corresponding value on the constants list:
            elif child < 0:
                self.children[i] = constants[-(child + 1)]

        
    def optimize(self, options):
        '''
        Method that optimizes the constants of a Tree using the given method (default is Nelder-Mead).

            Parameters:
                options (Options): Custom parameters of the algorithm execution.  
            
            Returns:
                Tree:              The optimized Tree.
        '''
        # We deepcopy the Tree because the optimization may not be succesful.
        tree = deepcopy(self)
        # We constantify the Tree.
        tree.constant_finder([], [], True)
        # Lambda function to be used on optimization.
        fun = lambda x: tree.error_calc_with_constants(options, x)
        try:
            # We perform the optimization.
            optim_res = optimize.minimize(fun, tree.constants, method = options.optim_method, options = {'maxiter': options.optim_max_iter, 'disp': False})
            # We undo and store the optimized constants.
            tree.undo_constants(optim_res.x)
            tree.constants = optim_res.x
            # We calculate the error and see if the expression is well defined.
            tree.error_calc(options)
            if isinstance(tree.error, float) and isfinite(tree.error):
                #const_tree.mutations['optimized'][0] += 1
                #const_tree.mutations['optimized'][1] += tree.error - self.error
                return tree
            else:
                return self
        except:
            return self

# ## FUNCTIONS ##
def init_problem(options):
    '''
    Function that formats some of the variables in order for the algorithm to work properly.

    Parameters:
        options (Options): Custom parameters of the algorithm execution.   
    '''
    # If there is only one variable on X, we convert [x1, x2, ... , xn] to [[x1], [x2], ... , [xn]] for the program to work.
    if not isinstance(options.data_X[0], list):
        options.data_X = [[x] for x in options.data_X]
    # We calculate the number of variables in the problem.
    options.n_var = len(options.data_X[0])
    if options.max_variables is None:
        options.max_variables = options.n_var
    # We load the database of functions to consider on our problem.
    # If the functions database was not customized, we load the whole database.
    if options.f_database == {}:
        options.functions = complete_functions_database
        # Names of functions.
        options.func_names = list(options.functions)
        # Weights of functions.
        options.func_weights = [value[3] for value in options.functions.values()]
    # Else, we load the especified functions.
    else:
        # We load the customized database and save the functions' names.
        options.functions = {func: complete_functions_database[func] for func in options.f_database}
        options.func_names = list(options.functions)
        # We check if the user also especified the weights.
        if isinstance(options.f_database, dict):
            options.func_weights = [options.f_database[func] for func in options.functions]
        # Else, we load the standard ones.
        else: 
            options.func_weights = [value[3] for value in options.functions.values()]
    # Table that organizes the functions on each mutation group.
    options.mut_groups = [
        [fun for fun, value in options.functions.items() if value[1] == it] for it in range(n_mut_groups)
                ]
    
def mutate(tree, temperature, complexity_dict, options):
    '''
    Method that mutates a tree. That is to say, it randomly selects a node between its descendants, then if it selected
    a Tree instance, it randomly selects a function in the same mutation group as the selected node's one and interchange them
    or if it selected a variable or constant, it interchanges it for another one.

        Parameters:
            tree (Tree):                     The tree to be mutate.
            temperature (float):             Annealing temperature.
            complexity_dict (dict):          Dictionary with the number of expressions at each complexity within the forest.
            options (Options):               Custom parameters of the algorithm execution.    
        Returns:
            copied_tree (Tree):              The mutated tree.
    '''
    # Default mutation weights (taken from PySR).
    mutation_weights = [0.048, 0.47, 0.79, 5.1, 1.7, 0.0020, 0.00023, 0.21, 0.4]
    # If the tree has many constants, we are more likely to mutate one of them. If it has none, we don't mutate any constant.
    mutation_weights[0] *= min(8, len(tree.constants)) / 8.0
    # If the expression has too many nodes, we won't add more.
    if tree.n_nodes >= options.complexity_limit - 1:
        mutation_weights[2] = 0
        mutation_weights[3] = 0
    # Counters to perform the mutations:
    attempts = 0
    succesful_mutation = False
    # We perform up to max_attempts mutations until one of them is succesful.
    while not succesful_mutation and attempts < options.max_attempts:
        # We randomly select a mutation case to perform.
        j = choices([1,2,3,4,5,6,7,8,9], mutation_weights)[0]
        # We deepcopy the Tree so we will be able to perform different mutations over the same Tree.
        copied_tree = deepcopy(tree)
        # Different mutations that may be implemented.
        match j:
            # Case 1: we perturb a constant of the expression.
            case 1:
                # We constantify the Tree because we are going to change one of its constants.
                copied_tree.constant_finder([], [], True)
                # We calculate the perturbation scale with the annealing temperature.
                a = (1 + options.constant_perturbation * temperature + options.minimum_perturbation) ** (2 * random() - 1)
                # The perturbation scale has 50% chance of being negative or positive.
                if not isinstance(a, complex):
                    if random() < 0.5:
                        a = -a
                    # We perturb one of the expression's constants.
                    copied_tree.constants[randint(0, len(copied_tree.constants) - 1)] *= a
                    try:
                        copied_tree.undo_constants(copied_tree.constants)
                        # We calculate the error of the mutated tree.
                        copied_tree.error_calc(options)
                        # If the tree is well defined, the mutation is succesful.
                        if isinstance(copied_tree.error, float) and isfinite(copied_tree.error) and len(copied_tree.variables) <= options.max_variables:
                            succesful_mutation = True
                            # We store the birth time of the expression.
                            copied_tree.birth = time.time() * 10 ** 7
                        else:
                            attempts += 1
                    except:
                        attempts += 1
            # Case 2: we change a node's function by another function with the same domain.
            case 2:
                # We have to make sure that we select a function of operator.
                selected_operator = False
                while not selected_operator:
                    # We call the node_chooser method with a randomly selected depth position.
                    parent, i = copied_tree.node_chooser(randint(1, copied_tree.n_nodes))
                    # If we selected the root node, it will always be a function or operator (by construction of the Trees).
                    if parent is None:
                        selected_operator = True
                        # We make sure that we do, in fact, change the operator.
                        #name = copied_tree.name
                        #while copied_tree.name == name:
                        # We select another function from its same mutation group and put it in the selected node one.
                        copied_tree.name = choices(options.mut_groups[options.functions[copied_tree.name][1]])[0]
                        copied_tree.func = options.functions[copied_tree.name][0]
                    # Else, we check if we selected a Tree.
                    elif isinstance(parent.children[i], Tree):
                        selected_operator = True
                        # We make sure that we do, in fact, change the operator:
                        #name = parent.children[i].name
                        #while parent.children[i].name == name:
                        # We select another function from its same mutation group and put it in the selected node one:
                        parent.children[i].name = choices(options.mut_groups[options.functions[parent.children[i].name][1]])[0]
                        parent.children[i].func = options.functions[parent.children[i].name][0]
                try:
                    # We calculate the error of the mutated tree.
                    copied_tree.error_calc(options)
                    # If the tree is well defined, the mutation is succesful.
                    if isinstance(copied_tree.error, float) and isfinite(copied_tree.error) and len(copied_tree.variables) <= options.max_variables:
                        succesful_mutation = True
                        # We store the birth time of the expression.
                        copied_tree.birth = time.time() * 10 ** 7
                    else:
                        attempts += 1
                except:
                    attempts += 1
            # Case 3: we prepend or append a node. Each one of the operations has a 50% chance of being made.
            case 3:
                # We prepend a node with 50% chance.
                if random() < 0.5:
                    # Loop to make sure that we generate a function with arity 2 or 3.
                    loop = True
                    while loop:
                        root = Tree(options, n_nodes = randint(2, 3))
                        if len(root.children) == root.n_nodes - 1: loop = False
                    # Now we add the tree to mutate to one of the leaves of the root.
                    root.children[randint(0, len(root.children)) - 1] = copied_tree
                    # We update some structural information of the Trees.
                    copied_tree.parent = root
                    root.n_nodes += copied_tree.n_nodes - 1
                    # We will return the root.
                    copied_tree = root
                # With 50% chance we append a node.
                else:
                    selected_terminal_node = False
                    # In order to append a node we have to select a terminal node.
                    while not selected_terminal_node:
                        # We call the node_chooser method with a randomly selected depth position.
                        parent, i = copied_tree.node_chooser(randint(1, copied_tree.n_nodes))
                        if parent is not None and not isinstance(parent.children[i], Tree):
                                selected_terminal_node = True
                    # Once we have a terminal node, we append an operator or function:
                    loop = True
                    while loop:
                        appendix = Tree(options, n_nodes = randint(2, 3))
                        if len(appendix.children) == appendix.n_nodes - 1: loop = False
                    # We append it:
                    parent.children[i] = appendix
                    # Now we have to update some of the Trees structural information:
                    appendix.parent = parent
                    descendants_update(parent, appendix.n_nodes - 1)
                try:
                    # We calculate the error of the mutated tree.
                    copied_tree.error_calc(options)
                    # We recalculate the constants of the Tree.
                    copied_tree.constant_finder([], [], False)
                    # If the tree is well defined, the mutation is succesful.
                    if isinstance(copied_tree.error, float) and isfinite(copied_tree.error) and len(copied_tree.variables) <= options.max_variables:
                        succesful_mutation = True
                        # We store the birth time of the expression.
                        copied_tree.birth = time.time() * 10 ** 7
                    else:
                        attempts += 1
                except:
                    attempts += 1  
            # Case 4: we insert a random node.
            case 4:
                # We make sure that we don't select the root node.
                selected_root = True
                while selected_root:
                    parent, i = copied_tree.node_chooser(randint(1, copied_tree.n_nodes))
                    if parent is not None:
                        selected_root = False
                        # Now we generate a random node to be inserted in the selected node position.
                        loop = True
                        while loop:
                            node = Tree(options, n_nodes = randint(2, 3))
                            if len(node.children) == node.n_nodes - 1: loop = False
                        node.children[randint(0, len(node.children) - 1)] = parent.children[i]                        
                        # We have to update some of the Trees structural data.
                        node.parent = parent
                        descendants_update(parent, node.n_nodes - 1)
                        if isinstance(parent.children[i], Tree):          
                            node.n_nodes += parent.children[i].n_nodes - 1
                            parent.children[i].parent = node
                        parent.children[i] = node
                try:
                    # We calculate the error of the mutated tree.
                    copied_tree.error_calc(options)
                    # We recalculate the constants of the Tree.
                    copied_tree.constant_finder([], [], False)
                    # If the tree is well defined, the mutation is succesful.
                    if isinstance(copied_tree.error, float) and isfinite(copied_tree.error) and len(copied_tree.variables) <= options.max_variables:
                        succesful_mutation = True
                        # We store the birth time of the expression.
                        copied_tree.birth = time.time() * 10 ** 7
                    else:
                        attempts += 1
                except:
                    attempts += 1
            # Case 5: we replace a node with a constant or variable.
            case 5:
                selected_root = True
                while selected_root:
                    parent, i = copied_tree.node_chooser(randint(1, copied_tree.n_nodes))
                    # We make sure that we did not select the root node (we don't want trees that are only constants).
                    if parent is not None:
                        selected_root = False
                        # We distinguish if the selected node is a Tree instance or not:
                        if isinstance(parent.children[i], Tree):
                            nodes_diff = 1 - parent.children[i].n_nodes
                            parent.children[i] = round(rd.normal(scale = options.constant_sd), randint(0, 15)) if random() < options.p_const_vs_var else randint(0, options.n_var - 1)
                            descendants_update(parent, nodes_diff)
                        else:
                            parent.children[i] = round(rd.normal(scale = options.constant_sd), randint(0, 15)) if random() < options.p_const_vs_var else randint(0, options.n_var - 1)
                try:
                    # We calculate the error of the mutated tree.
                    copied_tree.error_calc(options)
                    # We recalculate the constants of the Tree.
                    copied_tree.constant_finder([], [], False)
                    # If the tree is well defined, the mutation is succesful.
                    if isinstance(copied_tree.error, float) and isfinite(copied_tree.error) and len(copied_tree.variables) <= options.max_variables:
                        succesful_mutation = True
                        # We store the birth time of the expression.
                        copied_tree.birth = time.time() * 10 ** 7
                    else:
                        attempts += 1
                except:
                    attempts += 1  
            # Case 6: we simplify the tree.
            case 6:
                copied_tree = simplify(copied_tree, options)
                copied_tree.constant_finder([], [], False)
            # Case 7: we generate a completely new tree.
            case 7:
                copied_tree = Tree(n_nodes = randint(options.initial_complexity, options.complexity_limit), options = options)
                try:
                    # We calculate the error of the mutated tree.
                    copied_tree.error_calc(options)
                    # If the tree is well defined, the mutation is succesful.
                    if isinstance(copied_tree.error, float) and isfinite(copied_tree.error) and len(copied_tree.variables) <= options.max_variables:
                        succesful_mutation = True
                    else:
                        attempts += 1
                except:
                    attempts += 1 
            # Case 8: we don't mutate the tree.
            case 8:
                succesful_mutation = True
            # Case 9: we undo a composition. That is, we move the selected node a position up.
            case 9:
                # We have to make sure that we didn't select the root node.
                selected_root = True
                while selected_root:
                    parent, i = copied_tree.node_chooser(randint(1, copied_tree.n_nodes))
                    mutated = False
                    if parent is not None:
                        selected_root = False
                        # We distinguish if the selected node is a child of the root or not:
                        if parent.parent is not None:
                            # We search for the position of the parent inside its own parent:
                            for k in range(parent.parent.arity):
                                if parent.parent.children[k] is parent: break
                            # Now we distinguish if the selected node is a Tree instance or not:
                            if isinstance(parent.children[i], Tree):
                                diff = parent.children[i].n_nodes - parent.n_nodes 
                                parent.children[i].parent = parent.parent
                            else:
                                diff = 1 - parent.n_nodes
                            parent.parent.children[k] = parent.children[i]
                            mutated = True
                            descendants_update(parent.parent, diff)           
                        elif isinstance(parent.children[i], Tree):
                            copied_tree = parent.children[i]
                            copied_tree.parent = None
                            mutated = True
                if mutated:
                    try:
                        # We calculate the error of the mutated tree.
                        copied_tree.error_calc(options)
                        # We recalculate the constants of the Tree.
                        copied_tree.constant_finder([], [], False)
                        # If the tree is well defined, the mutation is succesful.
                        if isinstance(copied_tree.error, float) and isfinite(copied_tree.error) and len(copied_tree.variables) <= options.max_variables:
                            succesful_mutation = True
                            # We store the birth time of the expression.
                            copied_tree.birth = time.time() * 10 ** 7
                        else:
                            attempts += 1
                    except:
                        attempts += 1 
    if succesful_mutation:
        # We store the information of the mutation that was operated.
        #copied_tree.mutations[j][0] += 1
        #copied_tree.mutations[j][1] += copied_tree.error - tree.error
        # Now we calculate the annealing factor.
        try:
            q_annealing = math.exp(-(copied_tree.error - tree.error)/(options.annealing * temperature))
        except:
            q_annealing = math.inf
        # Finally we get the parsimony factor, calculated as a ratio of frequencies of the complexities inside the forest.
        C_1 = complexity_dict[tree.n_nodes]
        C_2 = complexity_dict[copied_tree.n_nodes]
        q_parsimony = C_1 / C_2 if C_2 != 0 else 1
        # Now we decide if we return the mutated expression or not.
        if random() < q_annealing * q_parsimony:
            return copied_tree, True
        else:
            return tree, False
    else:
        return tree, False

def node_calc(tree, counter):
    for child in tree.children:
        if isinstance(child, Tree):
            counter += node_calc(child, 0)
        else:
            counter += 1
    return counter + 1

def crossover(tree_A, tree_B, options):
    '''
    Method that crossovers two trees. That is to say, it randomly selects a branch of each tree and interchanges them.
    
    Parameters:
        tree_A (Tree):        First tree to crossover
        tree_B (Tree):        Second tree to crossover
        options (Options):    Custom parameters of the algorithm execution.
    Returns:
        copied_tree_A (Tree): First tree after the crossover
        copied_tree_B (Tree): Second tree after the crossover
    '''
    #L_1_A = tree_A.error
    #L_1_B = tree_B.error
    attempts = 0
    succesful_crossover = False
    while not succesful_crossover and attempts < options.max_attempts:
        copied_tree_A = deepcopy(tree_A)
        copied_tree_B = deepcopy(tree_B)
        try:
            # We select the branch from tree_A that will be cut:
            parent_A, i_A = copied_tree_A.node_chooser(randint(1, copied_tree_A.n_nodes - 1))
            # Notice that we do not take the whole tree as a branch. That's why we substract 1 from n_nodes:
            branch_A = deepcopy(parent_A.children[i_A])
            # We select the branch from tree_B that will be cut:
            parent_B, i_B = copied_tree_B.node_chooser(randint(1, copied_tree_B.n_nodes - 1))
            # Notice that we do not take the whole tree as a branch. That's why we substract 1 from n_nodes:
            branch_B = deepcopy(parent_B.children[i_B])
            # We make the crossover:
            parent_A.children[i_A] = branch_B
            parent_B.children[i_B] = branch_A
            # We have to update the number of descendants of each of the trees after crossover and the branches' parents:
            if isinstance(branch_A, Tree):
                if isinstance(branch_B, Tree):
                    difference = branch_B.n_nodes - branch_A.n_nodes
                    descendants_update(parent_A, difference)
                    descendants_update(parent_B, -difference)
                    branch_A.parent = parent_B
                    branch_B.parent = parent_A
                # If branch_B is a variable or constant, it number of descendants (including itself) will be 1:
                else:
                    difference = 1 - branch_A.n_nodes
                    descendants_update(parent_A, difference)
                    descendants_update(parent_B, -difference)
                    branch_A.parent = parent_B
            elif isinstance(branch_B, Tree):
                difference = branch_B.n_nodes - 1
                descendants_update(parent_A, difference)
                descendants_update(parent_B, -difference)
                branch_B.parent = parent_A
            # We don't take into account the case when both branch_A and branch_B are variables or constants
            # because in that case, the number of descendants of each tree do not change.
            # We calculate the error of the mutated tree.
            copied_tree_A.error_calc(options)
            copied_tree_B.error_calc(options)
            copied_tree_A.constant_finder([], [], False)
            copied_tree_B.constant_finder([], [], False)
            # If the tree is well defined, the mutation is succesful.
            if isinstance(copied_tree_A.error, float) and isfinite(copied_tree_A.error) and isinstance(copied_tree_B.error, float) and isfinite(copied_tree_B.error) and copied_tree_A.n_nodes <= options.complexity_limit and copied_tree_B.n_nodes <= options.complexity_limit and len(copied_tree_A.variables) <= options.max_variables and len(copied_tree_B.variables) <= options.max_variables:
                succesful_crossover = True
                copied_tree_A.birth = time.time() * 10 ** 7
                copied_tree_B.birth = time.time() * 10 ** 7
        except:
            pass
        attempts += 1
    if succesful_crossover:
        #copied_tree_A.mutations['crossover'][0] += 1
        #copied_tree_A.mutations['crossover'][1] += copied_tree_A.error - L_1_A
        #copied_tree_B.mutations['crossover'][0] += 1
        #copied_tree_B.mutations['crossover'][1] += copied_tree_B.error - L_1_B
        return copied_tree_A, copied_tree_B, True
    else:
        return tree_A, tree_B, False

def descendants_update(node, difference):
    '''
    Recursive function that, given a node inside a tree whose number of descendants changed, updates all
    the tree to match this difference of descendants.
    
        Parameters:
            node (Tree):      The node whose number of descendants varied
            difference (int): The variation of descendants the node suffered
    '''
    node.n_nodes += difference
    # If the node is inside a tree, we update its parent's number of descendats:
    if node.parent is not None:
        descendants_update(node.parent, difference)

def simplify(tree, options):
    '''
    Function that simplifies the expression of a given Tree.

        Parameters:
            tree (Tree):       The tree to be simplified.
            options (Options): Custom parameters of the algorithm execution.
    '''
    tree.children = [simplify(child, options) if isinstance(child, Tree) else child for child in tree.children]
    if tree.parent is not None and all([isinstance(child, float) for child in tree.children]):
        descendants_update(tree.parent, 1 - tree.n_nodes)
        return tree.func(*tree.children)
    # Here we will simplify
    # (constant + var) + constant = constant + var 
    # (constant * var) * constant = constant * var
    if tree.name in {'+', '*'} and tree.n_nodes == 5 and len(tree.constants) == 2 and ((isinstance(tree.children[0], Tree) and tree.children[0].name == tree.name) or (isinstance(tree.children[1], Tree) and tree.children[1].name == tree.name)):
        tree.children[0] = tree.variables[0]
        tree.children[1] = tree.func(*tree.constants)
        descendants_update(tree, -2)
        return tree
    # Here we will make some simplifications that rely on substraction.
    if tree.name == '-' and tree.n_nodes == 5 and len(tree.constants) == 2:
        if isinstance(tree.children[0], Tree) and tree.children[0].name == '-':
            descendants_update(tree, -2)
            # (var - constant) - constant = var + constant
            if isinstance(tree.children[0].children[0], int):
                tree.name = '+'
                tree.func = op.add
                tree.children[1] = - tree.children[1] - tree.children[0].children[1]
                tree.children[0] = tree.children[0].children[0]
            # (constant - var) - constant = constant - var
            else:
                tree.children[1] = tree.children[0].children[0] - tree.children[1]
                tree.children[0] = tree.children[0].children[1]
        elif isinstance(tree.children[1], Tree) and tree.children[1].name == '-':
            descendants_update(tree, -2)
            # constant - (constant - var) = constant + var
            if isinstance(tree.children[1].children[1], int):
                var = tree.children[1].children[1]
                tree.children[1] = tree.children[0] - tree.children[1].children[0]
                tree.children[0] = var
            # constant - (var - constant) = constant - var
            else:
                tree.name = '+'
                tree.func = op.add
                var = tree.children[1].children[0]
                tree.children[1] = tree.children[0] + tree.children[1].children[1]
                tree.children[0] = var
        return tree
    # Here we will simplify some compositions like
    # abs(abs(x)) = abs(x)
    # floor(floor(x)) = floor(x)
    # ceil(floor(x)) = floor(x), etc.
    if isinstance(tree.children[0], Tree) and tree.name in {'abs', 'floor', 'ceil'} and (tree.children[0].name == tree.name or tree.children[0].name == options.functions[tree.name][5]):
        if tree.parent is not None:
            tree.children[0].parent = tree.parent
            descendants_update(tree.parent, -1)
        else:
            tree.children[0].parent = None
            tree.children[0].error = tree.error
        return tree.children[0]
        
    # -(-x) = x and some inverse simplifications
    if isinstance(tree.children[0], Tree) and ((tree.name == '―' and tree.children[0].name == '―') or (tree.name in {'log', 'tan', 'log2', 'sinh', 'asinh', 'atanh'} and tree.children[0].name == options.functions[tree.name][5])):
        if tree.parent is not None:
            descendants_update(tree.parent, -2)
            if isinstance(tree.children[0].children[0], Tree):
                tree.children[0].children[0].parent = tree.parent
            return tree.children[0].children[0]
        elif isinstance(tree.children[0].children[0], Tree):
            tree.children[0].children[0].parent = None
            tree.children[0].children[0].error = tree.error
            return tree.children[0].children[0]
        
            
    # More simplifications that may be implemented:
    # 0 + x = x
    # 0 / x = 0
    # 0 * x = 0
    # 0 - x = -x
    # x ^ 0 = 1
    # x + x = 2*x
    # n*x + x = (n + 1) * x
    # (var / constant) / constant = var / constant
    # constant / (constant / var) = var / constant
    ##########
    # If no simplification was done, return the tree.
    return tree
    
if __name__ == '__main__':
    pass
