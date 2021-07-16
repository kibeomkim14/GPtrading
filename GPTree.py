# -*- coding: utf-8 -*-


import numpy.random as random
import pandas as pd
import numpy as np
from numpy.random import randint


#============================================================================#
#==============================BUILDING BLOCKS===============================#
#============================================================================#

# Define functions
def add(x,y): 
    if type(x) == 'pandas.Dataframe':
        x = x.iloc[-1]
    return x + y
        
def sub(x,y): 
    if type(x) == 'pandas.Dataframe':
        x = x.iloc[-1]
    return x - y

def mul(x,y): 
    if type(x) == 'pandas.Dataframe':
        x = x.iloc[-1]
    return x * y

def div(x,y):
    if type(x) == 'pandas.Dataframe':
        x = x.iloc[-1]
    if y == 0:
        return x
    return x / y

def norm(x,y): 
    if type(x) == 'pandas.Dataframe':
        x = x.iloc[-1]    
    return abs(x-y)

def max_(x,y): return max(x,y)
def min_(x,y): return min(x,y)
def avg_(x,y): return (x+y)/2

# Introduce Boolean functions
# NOTE: CHILD NODES ALWAYS HAVE TO BE RELATIONAL OR BOOLEAN.
def AND_(x,y) : return (x + y)/2 == 1
def OR_(x,y)  : return (x + y)/2 == 0.5
def NOT_(x,y) : return (x + y)/2 == 0 

# Introduce Relational functions
def geq(x,y) : return x > y
def leq(x,y) : return x < y
def eq(x,y)  : return x == y

# Introduce REAL FUNCTIONS which take real price and volume date into account.
def MAX_(data,n): 
    if type(data) == int:
        return max(data,n)
    else:
        return data.rolling(n).max().iloc[-1]
    
def MIN_(data,n): 
    if type(data) == int:
        return min(data,n)
    else:
        return data.rolling(n).min().iloc[-1]
    
def LAG_(data,n): 
    if type(data) == int:
        return 1
    else:
        return data.shift(n).iloc[-1]

def VOL_(data,n): 
    if type(data) == int:
        return 0
    else:
        return data.rolling(n).std().iloc[-1] * 100
    
    
def ROC_(data,n): 
    if type(data) == int:
        return 0
    else:
        return ((data/data.shift(n))-1).iloc[-1] * 100
def SMA_(data,n):
    if type(data) == int:
        return 0
    else:
        return data.rolling(n).mean().iloc[-1]

def EMA_(data,n):
    if type(data) == int:
        return 0
    else:
        return data.ewm(n).mean().iloc[-1]

## BUILD POPULATION AND TERMINALS
FUNCTIONS   = [add,sub,mul,div,norm,max_,min_,avg_]
REALFUNC    = [MAX_,MIN_,LAG_,VOL_,ROC_,SMA_,EMA_]
RELATIONALS = [geq,leq,eq]

BOOLEANS    = [AND_,OR_,NOT_]
REALVALUE   = ['P', 'V']
TERMINALS   = []
for i in range(0,250): TERMINALS.append(i) 
            
#============================================================================#
#============================================================================#
#============================================================================#

P_Mutation  = 0.1
P_Crossover = 0.6

##############################################################################
############################ BUILD GP TREE CLASS #############################
##############################################################################


class Tree:
    
    def __init__(self, data = None, left = None, right = None, depth = 0):
        self.f       = data  # function or terminal value is assigned
        self.left    = left
        self.right   = right     
        self.depth   = depth # denotes the depth level
        self.name    = None  # contains the name of terminal or function
        self.m_depth = None
        

    def make_(self, max_depth, days, grow=False, prev_func = None):                          
        '''
        creates random tree using either grow or full method upon user's choice 
        For the description of grow and full method, see KOZA I, II
        '''
        
        random.seed()
        self.m_depth = max_depth
        
        if self.depth == 0:  
            self.f = BOOLEANS[randint(0, len(BOOLEANS))] # At the root, we only assign boolean functions so the outcome is always 1 or 0   

        elif self.depth == 1: # At depth = 1, we assign relationals or boolean functions.
            self.f = RELATIONALS[randint(0, len(RELATIONALS))]  
                
        elif self.depth == self.m_depth:   
            if prev_func in REALFUNC:
                self.f = REALVALUE[randint(0, len(REALVALUE))] # terminals are assigned at leaves
            else:
                # terminals are assigned at leaves
                self.f = randint(1,days) 
        else:
        # If grow is false (i.e full method), intermediate nodes will always have real functions. 
        # If grow is True, it will have 50% chance of getting the node as terminal
            if grow and (random.random() > 0.5):                  
                self.f = randint(1,days) 
            else:
                if self.depth == self.m_depth -1 and (random.random() > 0.5):
                    self.f = REALFUNC[randint(0, len(REALFUNC))]
                else:
                    self.f = FUNCTIONS[randint(0, len(FUNCTIONS))]
     
        
        # After assgining function or terminal, set the node name.        
        if (type(self.f) != int) and (self.f not in REALVALUE):
            self.name = self.f.__name__
        else: 
            self.name = str(self.f)     
        
        
        # IF current node is not a value, then we apply make_ recursively 
        # until the last node contains a value in terminal.
        if (type(self.f) != int) and (self.f not in REALVALUE):
            
            self.left  = Tree(depth =self.depth + 1)          
            self.right = Tree(depth =self.depth + 1)
            
            self.left.make_(self.m_depth, days, grow, prev_func = self.f)            
            self.right.make_(self.m_depth, days, grow)


    def calc_(self, Price,Volume): 
        '''
        calc_ function can be used only after Tree.make_() is performed
        the function gives final value of the tree.
        '''
        
        # At root, print the final calculation.
        if self.depth == 0:
            return self.f(self.left.calc_(Price,Volume), self.right.calc_(Price,Volume))
        
        # If we are not at root, we compute trees.
        # calculate the leaves first if current node is a function
        if (type(self.f) != int) and (self.f not in REALVALUE): 
            return self.f(self.left.calc_(Price,Volume), self.right.calc_(Price,Volume))
        # we take real price as terminal
        elif self.f == 'P':
            return Price
        # we take real volume as terminal
        elif self.f == 'V':
            return Volume
        # Returns terminals other than p or V
        else: 
            return self.f 
   
    
    def print_(self, string = ""):  
        '''
        Prints the structure of the tree.
        At each node, this function prints the depth no. and the type of function or terminal.
        '''
        
        print("{}Depth {}: {}".format(string, self.depth, self.name))        
        
        if self.left:  
        
            self.left.print_(string + "               ")
        
        if self.right: 
                
            self.right.print_(string + "               ")

    def nodes_(self):
        '''
        returns the number of leaves (i.e. terminals at the bottom).
        returns 1 if the node is the terminal node
        '''
        if self.f not in REALFUNC:
            return 1
        
        # if the node of interest has left and right child nodes, 
        # calculate its branches recursively.
        if self.left:
            l = self.left.nodes_()
        if self.right:
            r = self.right.nodes_()
            
        return 1 + l + r

    def mutation_(self,days,grow=False):
        '''
        The idea of mutation is generating a newly formed tree at the point
        where the probability is higher than the threshold, P_Mutation
        unary operator aimed at restoring diversity in a population by applying random
        modifications to individual structures.
        '''
        
        if random.random() < P_Mutation: 
            self.make_(self.m_depth,days,grow)
            
            try:
                p =  pd.Series(np.random.randn(500))
                v =  pd.Series(np.random.randn(500))
                self.calc_(p, v)
            except AttributeError:
                pass
            
            
        elif self.left:
            self.left.mutation_(days,grow)
        elif self.right:
            self.right.mutation_(days,grow) 

            
    def copy_(self): 
        # This function copies the whole tree 
        # and returns as Tree object.
        
        # Initialize new tree
        t        = Tree()
        
        # Copy and paste the existing tree attributes
        t.f      = self.f
        t.left   = self.left
        t.right  = self.right
        t.depth  = self.depth
        t.name   = self.name
        t.m_depth= self.m_depth

        # Excute the same process recursively for both branches.
        if self.left:  
            t.left  = self.left.copy_()
        if self.right: 
            t.right = self.right.copy_()
        return t
    
    
    def paste_(self,other):
        # copies all attributes that other tree have.
        t1 = other.copy_()
        
        self.f      = t1.f
        self.left   = t1.left
        self.right  = t1.right
        self.depth  = t1.depth
        self.name   = t1.name
        self.m_depth= t1.m_depth

        # Excute the same process recursively for both branches.
        if t1.left:  
            self.left.paste_(t1.left)
        if t1.right: 
            self.right.paste_(t1.right)


    def crossover_(self,other,crossPt=None): 
        '''
        Swap parts of 2 trees at random nodes. First, we set crossover point
        for both parent trees then perform swap. If the output trees are not
        compatible - i.e. if we can't calculate, then we do not use these
        output trees.
        '''
        
        if self.depth == 0:
            crossPt = randint(self.m_depth) 
        
        # If crossPt is same as the node depth, perform crossover.
        if crossPt == self.depth and random.random() > P_Crossover:
            t1 = self.copy_()
            t2 = other.copy_()
        
            self.paste_(t2)
            other.paste_(t1)
            
            try:
                # check whether output tree is valid. if not,
                # pass this function.
                p =  pd.Series(np.random.randn(500))
                v =  pd.Series(np.random.randn(500))
                self.calc_(p, v)
                other.calc_(p, v)
                
            except AttributeError:
                self.paste_(t1)
                other.paste_(t2)
                pass
                
            
        # If we perform crossover function on grow and full tree, then both 
        # trees are less likely to have sane swap points. If this happens, we
        # cannot use the output tree for next run.
        # Hence, we don't perform crossover and attempts next pair of trees.
        try:
            if random.random() < 0.5:
                self.left.crossover_(other.left,crossPt)
            else:
                self.right.crossover_(other.right,crossPt)
        except AttributeError:
            pass
