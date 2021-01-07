import functools
import numpy as np


class Metrices():
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.pix_stat =  np.zeros((grid_size,grid_size,2))

    def update_pix(self, row, col, status):
        self.pix_stat[row][col][0 if status > 0 else 1] += 1

def analyze(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # print(f'Run {func.__name__} with args={args}')
        return func(*args, **kwargs)
    return wrapper


def print_stats(metrices: Metrices):
    print(f'Printing summary of stats ->')

    for row in range(metrices.grid_size):
        for col in range(metrices.grid_size):
            success_count = metrices.pix_stat[row][col][0]
            fail_count = metrices.pix_stat[row][col][1]
            print(f'Row {row}, Col {col}: Success={success_count}, Fail={fail_count}')

    # Most 10 rows
    top = []
    print(type(metrices.pix_stat))    
    _max = np.argmax(metrices.pix_stat, axis=2)
    print(_max)
    #top = max(metrices.pix_stat, key=lambda i,j: metrices[i][j])

    # Most 10 cols
    botoom = []