# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:41:56 2021

@author: kibeom
"""

import time
import GPTree
import numpy  as np
import pandas as pd
import multiprocessing as mp

#============================================================================#
#=============================== FUNCTIONS ==================================#
#============================================================================#


def grow_(max_depth,termninal_max):
    '''
    A function makes grow-method tree, given parameters.

    Input:
    max_depth(int)       - depth of output tree
    terminal_max(int)    - window interval for the program to apply GPTree
    
    Output:
    tree_grow(Tree)      - grow-method tree
    '''   
    
    tree_grow = GPTree.Tree()
    tree_grow.make_(max_depth,termninal_max,grow = True)
    return tree_grow

def full_(max_depth,termninal_max):
    '''
    A function makes full-method tree, given parameters.

    Input:
    max_depth(int)       - depth of output tree
    terminal_max(int)    - window interval for the program to apply GPTree
    
    Output:
    tree_full(Tree)      - full-method tree
    '''   
    
    tree_full = GPTree.Tree()
    tree_full.make_(max_depth,termninal_max,grow = False)
    return tree_full


def performance(tree,price,volume,days):
    '''
    A function calculates annual return of a tree as an
    input. To calculate the return, the user need to input price and volume data
    to obtain returns as an output.

    Input:
    tree(Tree)           - a tree
    price(pd.DataFrame)  - price data 
    volume(pd.DataFrame) - volume data
    days(int)            - window interval for the program to apply GPree

    Output
    annual_return(float) - annual return in %(including initial investment)
    annual_vol(float)    - annual return volatility in %
    weight(pd.DataFrame) - buy/sell signal of the Tree during the tested period
    '''

    signal  = []
    sig_sum = 0

    for i in range(len(price) - days + 1):
        signal.append(tree.calc_(price.iloc[i:i+days],volume.iloc[i:i+days])) 
        sig_sum += signal[-1]

    # if signal generated does not have any buy signal, no investment made.
    # hence, return 1,0,0 indicating no return, no volatility.
    if sig_sum == 0:
        return 1
    
    # Otherwise, generate signal dataframe.
    weight = pd.DataFrame(signal, index = price.index[days-1:len(price)], columns=['signal']) * 1
    #print(weight)
    price  = price[weight.index]

    pos = 0 # position
    cum_return = 1 # cumulative return
    # ret_set = [] # list of returns which will be appended soon.

    for i in range(1,weight.shape[0]):
        # Given that no position (pos =0) and buy signal at the date,
        # enter a long position. (pos =1)
        if pos == 0 and weight.iloc[i].values[0] == 1:
            pos += 1
        elif pos == 1:
            # As long as we are in the LONG position, return is generated
            # until we sell the asset. 
            if weight.iloc[i].values[0] == 0:
                pos = 0
            
            ret = price.iloc[i]/price.iloc[i-1]
            cum_return *= ret # update cumulative return by multiplying.
            # ret_set.append(ret-1) 
    
    # annualize cum_return to get an annual return
    annual_return = cum_return ** (252/weight.shape[0]) 
    
    # with the ret_set we obtained above, we calculate daily return volatility
    # Then, annualize the daily volatility.
    # annual_vol    = np.std(ret_set)*np.sqrt(252)
    
    return annual_return

def selection(population, returns, BM_return):
    '''
    A function calculates fitness of all programs in population then calculates
    selection probabilities. After that, a tree is randomly selected given those 
    probabilities and returns a tree as an output.

    Input:
    population(list) - list of trees from population
    returns(list)    - annual returns of trees from population
    BM-return(float) - BM return value which will be used to calculate an excess return

    Output
    selected(Tree) - a tree will be selected after fitness evaluation and selection process 
    best(Tree)     - a tree with the best fitness
    '''
    
    # Calculate raw fitness, which is an excess return compared to buy-hold strategy on SP500 index.
    raw_fitness = np.array(returns) - BM_return
    
    # Get maximum and minimum of fitness values
    Mx = max(raw_fitness)
    Mn = min(raw_fitness)

    # Perform rank-based method selection accroding to Baker, Whitley. 
    rank    = abs(raw_fitness.argsort().argsort() - len(raw_fitness))
    fitness = Mx - (Mx - Mn) * (rank -1)/(len(rank)-1)
    
    # Since, finess could be negative, we adjust so that all fitnesses are positive.
    adj_fitness = fitness + abs(min(fitness))
    
    # Now, calculate probability of selection for all programs
    if sum(adj_fitness) < 0.05:
        P_select = np.full(len(population),1/len(adj_fitness))
    else:
        P_select = adj_fitness/np.sum(adj_fitness)

    # Given the selection probabiltiy, we select program.
    selected_idx = np.random.choice(rank, 1, p = P_select)[0]
    
    # returns tree selected, best tree 
    return population[np.where(rank==selected_idx)[0][0]], population[np.where(rank==1)[0][0]]




#============================================================================#
#============================= DATA SETTING =================================#
#============================================================================#

# Import S&P future data
data = pd.read_csv('S&Pctsfuture.csv',index_col='Date')
data = data.iloc[::-1] # Reverse dataframe
data = data.dropna()

vol   = data['Volume'] 
close = data['Close/Last'] 

#============================================================================#
#=============================  PARAMETERS  =================================#
#============================================================================#

# initial population parameters
pop_size    = 500
depth       = 3
term_max    = 21 # 3 month window (given 252 trading days in a year)


best_tree   = None
best_return = 0
best_gen    = 0
GENERATIONS = 50

# Set Benchmark annual returns to measure performance
# See SPindex csv file for more detail.

BM_ret_tr     = 1.10934558317888
BM_ret_te     = 1.150587370026509


'''
Since we will be executing GP algorithm for large set of trees and the computation
time increases drastically. Without parallel programming, running the below code
takes more than 30 minutes. Hence, we import multiprocessing module to enable
parallel programming. 
'''

if __name__ == '__main__':
    
    start = time.time()
    t1    = start
    
    # Initialize multiprocessing pool with 4 CPUs
    pool  = mp.Pool(processes=4)



    #============================================================================#
    #==============================   INITIALIZE   ==============================#
    #============================================================================#

    # Initialize population of trees using multiprocessing module
    init_pool = []  
    for i in range(pop_size//2):
        init_pool.append(pool.apply_async(grow_, args=(depth,term_max)))
        init_pool.append(pool.apply_async(full_, args=(depth,term_max)))
    
    population = [p.get() for p in init_pool]
    
    # Measure time taken
    t2 = time.time()
    print('COMPLETE - Initialize population : {s:.3f} seconds'.format(s=t2-t1))
    t1 = t2

    #============================================================================#
    #==============================   TRAINING   ================================#
    #============================================================================#
    
    
    for gen in range(GENERATIONS):    
        if best_gen + 15 < gen:
            print("Best tree not found for 10 generations. Breaking the loop..")
            break
        print("-------- Gen {} Start.".format(gen+1))
        
        # Performance of trees for the period from 2012-04-19 ~ 2016-04-22       
        perf = []
        for tree in population:
            perf.append(pool.apply_async(performance, args=(tree,close[1500-term_max:],vol[1500-term_max:],term_max)))
    
        annuals = [p.get() for p in perf]
              
        # Apply selection process given performance of trees in population
        next_population = []
        for i in range(pop_size):
            parent1, gen_best = selection(population, annuals, BM_ret_tr)
            parent2, _        = selection(population, annuals, BM_ret_tr)
            
            parent1.crossover_(parent2)
            parent1.mutation_(term_max,True)        
            next_population.append(parent1)
                
        
        population = next_population
        a_ret = performance(gen_best,close[1500-term_max:],vol[1500-term_max:],term_max)
        
        if a_ret > best_return:
            best_return = a_ret
            best_gen    = gen
            best_tree   = gen_best.copy_()
            print("________________________")
            print("gen:", gen+1, ", best_annual_return:", round(a_ret,4)) 
            best_tree.print_()
        
        # Measure time taken
        t2 = time.time()
        print('COMPLETE - Generation {g} : {s:.3f} seconds'.format(g=gen+1, s=t2-t1))
        t1 = t2
        
    
    
    #============================================================================#
    #===========================   TRAIN SUMMARY   ==============================#
    #============================================================================#
    
    
    print("\n\n_________________________________________________\nEND OF RUN\n best tree obtained at generation " + str(best_gen+1) +\
          " and has Return=" + str(round(best_return,3)))
    best_tree.print_()
    
    end = time.time()
    print('RUN COMPLETE : {} min {} seconds'.format(round((end - start)//60) , round(((end - start)%60))))
    '''
    sig  = []
    sig_sum = 0 
    for i in range(len(close[250:1250])):
        sig.append(tree.calc_(close[250+i-term_max:250+i],vol[250+i-term_max:250+i])) 
        sig_sum += sig[-1]
        
    if sig_sum == 0:
        print('no weight..')
    
    best_w = pd.DataFrame(sig, index = close[250:1250].index, columns=['signal']) * 1
    
    
    '''