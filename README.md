# fad-2020-2021

Dataset available [here](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017)

Probably questions to answer

Who is the best team of all time
Which teams dominated different eras of football
What trends have there been in international football throughout the ages - home advantage, total goals scored, distribution of teams' strength etc
Can we say anything about geopolitics from football fixtures - how has the number of countries changed, which teams like to play each other
Which countries host the most matches where they themselves are not participating in
How much, if at all, does hosting a major tournament help a country's chances in the tournament
Which teams are the most active in playing friendlies and friendly tournaments - does it help or hurt them
World Cup 2022 Winner? (check programma)


## Setup venv

Create virtual environment

`python3 -m venv venv`

Activate venv

`source venv/bin/activate`

## Setup numba

After installed Anaconda:

`conda install numba & conda install cudatoolkit`

Allow to use the numba.jit decorator for the function we want to compute over the GPU. 
However, if CPU is passed as an argument then the jit tries to optimize the code run faster on CPU and improves the speed too.

```
from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer   
  
# normal function to run on cpu
def func(a):                                
    for i in range(10000000):
        a[i]+= 1      
  
# function optimized to run on gpu 
@jit(target ="cuda")                         
def func2(a):
    for i in range(10000000):
        a[i]+= 1
if __name__=="__main__":
    n = 10000000                            
    a = np.ones(n, dtype = np.float64)
    b = np.ones(n, dtype = np.float32)
      
    start = timer()
    func(a)
    print("without GPU:", timer()-start)    
      
    start = timer()
    func2(a)
    print("with GPU:", timer()-start)
```