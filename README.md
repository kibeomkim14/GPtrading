# Genetic Programming for Trading

__`Team Members`__: Kibeom Kim 


## Setup 
To start our project you must have the following libraries installed on your computer.

1. Numpy
2. time
3. pandas
4. multiprocessing

To avoid package dependecies, I recommend you to setup a virtual python environment. After the setup, install above 
packages by running ``requirements.pip`` file on your terminal like:

```
pip install -r requirements.pip
```
## Project Goal
Goal of this project is to attempt to beat the buy & hold strategy(B&H) on S&P500 Futures return using genetic programming(GP). 
GP is a technique of evolving programs, starting from a population of unfit (usually random) programs, fit for a particular
task by applying operations analogous to natural genetic processes to the population of programs. It is a heuristic search 
technique that searches for an optimal (or at least suitable) program among the space of all programs.

At the start, the terminals and population (in other words, programs) will be specified and will be fed into the GP algorithm.
As it goes, the algorithm will be optimized in the sense of maximizing investment returns - parameters associated with GP 
algorithm will be adjusted to achieve such return. For more information on the algorithm and its use, please look at 
Potvin, Soriano and Vallee (2004).

Specifics of the project are listed as below:


• Asset name: E-Mini S&P 500 (Continuous contract)

• Benchmark : E-Mini S&P 500 B&H

• Terminal  : arbitrary constants, price, volume, boolean (True,False), k-days in [0,250]

• Population: +, -, *, / , max(), min(), boolean operators (AND, OR, NOT), relational operators
(≤ , ≥, =), lag-operator (e.g. using k-day lagged price or volume), moving average (MA),
exponential moving average(EMA) and other real-functions (such as Rate of Change(ROC),
k-day volatility(Vol) and more) will be added throughout the project.

• Time-frame: Daily OHLCV(Open, High, Low, Close, Volume) data in the range 2011/04/20 -
2021/04/19 will be used. (10 Years)

• Performance metrics: the performance of the strategy will be measured in annualized excess
return(compared to the Benchmark) and annualized Sharpe Ratio.


## Structure of Repository
```
├── README.md
├── data
│   └── original_data /S&Pctsfuture.csv
└── src
    ├── GPTree.py
    ├── func.py
    └── main.py
```
