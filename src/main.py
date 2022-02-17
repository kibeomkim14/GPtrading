# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:41:56 2021

@author: kibeom
"""

import time
import GPTree
import numpy as np
import pandas as pd
import multiprocessing as mp
from func import grow_, full_, performance, selection

#============================================================================#
#============================= DATA SETTING =================================#
#============================================================================#

# Import S&P future data
data = pd.read_csv('../data/S&Pctsfuture.csv', index_col='Date')
data = data.iloc[::-1] # Reverse dataframe
data = data.dropna()

vol   = data['Volume'] 
close = data['Close/Last'] 

#============================================================================#
#=============================  PARAMETERS  =================================#
#============================================================================#

# initial population parameters
pop_size = 500
depth    = 3
term_max = 21 # 3 month window (given 252 trading days in a year)


best_tree   = None
best_return = 0
best_gen    = 0
GENERATIONS = 50

# Set Benchmark annual returns to measure performance
# Return is calculated then annualized.
# See SPindex.csv file for more detail.

BM_ret_tr = 1.10934558317888
BM_ret_te = 1.150587370026509


'''
Since we will be executing GP algorithm for large set of trees, the computation
time associated with this task increases drastically. Without parallel programming, 
running the below code takes more than 30 minutes. Hence, we import multiprocessing 
module to enable parallel programming to speed up the computation. 
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

    #============================================================================
    #==============================   TRAINING   ================================
    #============================================================================

    for gen in range(GENERATIONS):    
        
        # if best tree is not found within 15 generations, finish the process.
        if best_gen + 15 < gen:
            print("Best tree not found for 15 generations. Breaking the loop..")
            break
        print("-------- Gen {} Start.".format(gen+1))
        
        # Performance of trees for the training period       
        perf = []
        for tree in population:
            # assess the performance (annual return)
            perf.append(pool.apply_async(performance, args=(tree,close[1500-term_max:],vol[1500-term_max:],term_max)))
    
        annuals = [p.get() for p in perf]
              
        # Apply selection process given performance of trees in population
        next_population = []
        for i in range(pop_size):
            parent1, gen_best = selection(population, annuals, BM_ret_tr)
            parent2, _ = selection(population, annuals, BM_ret_tr)
            
            # After selecting two trees, perform crossover and mutation operation according to Koza.
            parent1.crossover_(parent2)
            parent1.mutation_(term_max,True)        
            next_population.append(parent1)

        population = next_population
        
        # calculate the return for best tree from each generation.
        a_ret = performance(gen_best,close[1500-term_max:],vol[1500-term_max:],term_max)
        
        # Compare the generation best with all-time best.
        # If generation best is better, assign the tree as all-time best.
        # Then print the result.
        if a_ret > best_return:
            best_return = a_ret
            best_gen  = gen
            best_tree = gen_best.copy_()
            print("________________________")
            print("gen:", gen+1, ", best_annual_return:", round(a_ret,4)) 
            best_tree.print_()
        
        # Measure time taken for each generation.
        t2 = time.time()
        print('COMPLETE - Generation {g} : {s:.3f} seconds'.format(g=gen+1, s=t2-t1))
        t1 = t2

    #============================================================================#
    #===========================   TRAIN SUMMARY   ==============================#
    #============================================================================#
    print("\n\n_________________________________________________\nEND OF RUN\n best tree obtained at generation " \
          + str(best_gen+1) + " and has Return=" + str(round(best_return,3)))
    best_tree.print_()
    end = time.time()
    print('RUN COMPLETE : {} min {} seconds'.format(round((end - start)//60) , round(((end - start)%60))))

    # Obtain signal based on the rule of best tree generated.
    sig  = []
    sig_sum = 0 
    for i in range(len(close[250:1250])):
        sig.append(tree.calc_(close[250 + i - term_max:250 + i], vol[250 + i - term_max:250 + i]))
        sig_sum += sig[-1]
        
    if sig_sum == 0:
        print('no weight..')
    
    # store the asset allocation weight as pd.DataFrame.
    best_w = pd.DataFrame(sig, index = close[250:1250].index, columns=['signal']) * 1
    
    
