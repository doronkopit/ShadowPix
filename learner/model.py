import functools
import numpy as np


class PixModel():
    def __init__(self, grid_size, neighbors_update=True, radius=1):
        self.grid_size = grid_size

        # The minimum score for a pixel
        self.bias = 0.1

        # The values for success and fail updates of a pixel
        self.gain = 0.5
        self.punish = -0.15

        # The values for success and fail of neighboring pixels
        self.neighbors_update = neighbors_update
        self.neighbor_factor = 0.07
        self.radius = radius

        # [success_count, fail_count, neighbor_succ_count, neighbor_fail_count] for each pixel
        self.pix_stat =  np.zeros((grid_size, grid_size, 4))

        # score for each pixel, initially 1
        self.pix_score = np.full((grid_size, grid_size), self.bias)
        
        # total score = number of pixels
        self.total_score = grid_size * grid_size * self.bias 

    def update_pix(self, row, col, status):
        self.pix_stat[row][col][0 if status > 0 else 1] += 1

        update = self.gain if status > 0 else self.punish
        self.__make_update(row, col, update)

        if self.neighbors_update:
            self.update_neighbors(row, col, update)
    
    def __make_update(self, row, col, update):
        original_pix_score = self.pix_score[row][col]
        updated_pix_score = original_pix_score + update
        updated_pix_score = self.bias if updated_pix_score < self.bias else updated_pix_score 
        self.pix_score[row][col] = updated_pix_score
        self.total_score += (updated_pix_score - original_pix_score)
    
    def update_neighbors(self, row, col, update):
        update = update * self.neighbor_factor 
        for i in range(row-self.radius, row+self.radius+1):
            for j in range(col-self.radius, col+self.radius+1):
                if (i,j) == (row, col) or self.is_out_of_bounds(i, j):
                    continue
                self.pix_stat[row][col][2 if update > 0 else 3] += 1
                self.__make_update(i, j, update)

    def is_out_of_bounds(self, row, col):
        def is_index_out_of_bounds(index):
            return index < 0 or index >= self.grid_size
        return is_index_out_of_bounds(row) or is_index_out_of_bounds(col)
        
def analyze(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # print(f'Run {func.__name__} with args={args}')
        return func(*args, **kwargs)
    return wrapper
