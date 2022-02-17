import numpy as np
import pandas as pd
from GPTree import Tree


# ============================================================================#
# =============================== FUNCTIONS ==================================#
# ============================================================================#


def grow_(max_depth, terminal_max):
    """
    A function makes grow-method tree, given parameters.

    Parameters:
        max_depth(int)       - depth of output tree
        terminal_max(int)    - window interval for the program to apply GPTree

    Returns:
        tree_grow(Tree)      - grow-method tree
    """
    tree_grow = Tree()
    tree_grow.make_(max_depth, terminal_max, grow=True)
    return tree_grow


def full_(max_depth, terminal_max):
    """
    A function makes full-method tree, given parameters.

    Parameters:
        max_depth(int)       - depth of output tree
        terminal_max(int)    - window interval for the program to apply GPTree

    Returns:
        tree_full(Tree)      - full-method tree
    """

    tree_full = Tree()
    tree_full.make_(max_depth, terminal_max, grow=False)
    return tree_full


def performance(tree, price, volume, days):
    """
    A function calculates annual return of a tree as an
    input. To calculate the return, the user need to input price and volume data
    to obtain returns as an output.

    Parameters:
        tree(Tree)           - a tree
        price(pd.DataFrame)  - price data
        volume(pd.DataFrame) - volume data
        days(int)            - window interval for the program to apply GPree

    Returns:
        annual_return(float) - annual return in %(including initial investment)
        annual_vol(float)    - annual return volatility in %
        weight(pd.DataFrame) - buy/sell signal of the Tree during the tested period
    """

    signal = []
    sig_sum = 0

    for i in range(len(price) - days + 1):
        signal.append(tree.calc_(price.iloc[i:i + days], volume.iloc[i:i + days]))
        sig_sum += signal[-1]

    # if signal generated does not have any buy signal, no investment made.
    # hence, return 1,0,0 indicating no return, no volatility.
    if sig_sum == 0:
        return 1

    # Otherwise, generate signal dataframe.
    weight = pd.DataFrame(signal, index=price.index[days - 1:len(price)], columns=['signal']) * 1
    # print(weight)
    price = price[weight.index]

    pos = 0  # position
    cum_return = 1  # cumulative return
    # ret_set = [] # list of returns which will be appended soon. (Optional)

    for i in range(1, weight.shape[0]):
        # Given that no position (pos =0) and buy signal at the date,
        # enter a long position. (pos =1)
        if pos == 0 and weight.iloc[i].values[0] == 1:
            pos += 1
        elif pos == 1:
            # As long as we are in the LONG position, return is generated
            # until we sell the asset.
            if weight.iloc[i].values[0] == 0:
                pos = 0

            ret = price.iloc[i] / price.iloc[i - 1]
            cum_return *= ret  # update cumulative return by multiplying.
            # ret_set.append(ret-1)

    # annualize cum_return to get an annual return
    annual_return = cum_return ** (252 / weight.shape[0])

    # with the ret_set we obtained above, we calculate daily return volatility
    # (Optional, needed for calculating Sharpe ratio)Then, annualize the daily volatility.
    # annual_vol    = np.std(ret_set)*np.sqrt(252)

    return annual_return


def selection(population, returns, BM_return):
    """
    A function calculates fitness of all programs in population then calculates
    selection probabilities. After that, a tree is randomly selected given those
    probabilities and returns a tree as an output.

    Parameters:
        population(list) - list of trees from population
        returns(list)    - annual returns of trees from population
        BM-return(float) - BM return value which will be used to calculate an excess return

    Returns:
        selected(Tree) - a tree will be selected after fitness evaluation and selection process
        best(Tree)     - a tree with the best fitness
    """

    # Calculate raw fitness, which is an excess return compared to buy-hold strategy on SP500 index.
    raw_fitness = np.array(returns) - BM_return

    # Get maximum and minimum of fitness values
    Mx = max(raw_fitness)
    Mn = min(raw_fitness)

    # Perform rank-based method selection according to Baker, Whitley.
    rank = abs(raw_fitness.argsort().argsort() - len(raw_fitness))
    fitness = Mx - (Mx - Mn) * (rank - 1) / (len(rank) - 1)

    # Since, fitness could be negative, we adjust so that all fitness are positive.
    adj_fitness = fitness + abs(min(fitness))

    # Now, calculate probability of selection for all programs
    if sum(adj_fitness) < 0.05:
        P_select = np.full(len(population), 1 / len(adj_fitness))
    else:
        P_select = adj_fitness / np.sum(adj_fitness)

    # Given the selection probability, we select program.
    selected_idx = np.random.choice(rank, 1, p=P_select)[0]

    # returns tree selected, best tree
    return population[np.where(rank == selected_idx)[0][0]], population[np.where(rank == 1)[0][0]]


