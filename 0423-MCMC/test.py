from multiprocessing import Pool
import numpy as np

def funtion(x):
    print(x*x)

if __name__ == '__main__':
    with Pool(30) as p:
        p.map(funtion, np.arange(10))
