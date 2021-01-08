import functools
import numpy as np


class Metrices():
    def __init__(self, grid_size):
        self.grid_size = grid_size

        self.bias = 0.1
        self.gain = 0.5
        self.punish = -0.1

        # [success_count, fail_count] for each pixel
        self.pix_stat =  np.zeros((grid_size,grid_size,2))

        # score for each pixel, initially 1
        self.pix_score = np.full((grid_size, grid_size), self.bias)
        
        # total score = number of pixels
        self.total_score = grid_size * grid_size * self.bias 

    def update_pix(self, row, col, status):
        self.pix_stat[row][col][0 if status > 0 else 1] += 1

        update = self.gain if status > 0 else self.punish

        original_pix_score = self.pix_score[row][col]
        updated_pix_score = original_pix_score + update
        updated_pix_score = self.bias if updated_pix_score < self.bias else updated_pix_score 
        self.pix_score[row][col] = updated_pix_score
        self.total_score += (updated_pix_score - original_pix_score)
         

def analyze(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # print(f'Run {func.__name__} with args={args}')
        return func(*args, **kwargs)
    return wrapper
