from global_method import GlobalMethod
from learner.analyze_tools import analyze, Metrices
import numpy as np


class GlobalMethodLearner(GlobalMethod):
    def __init__(self, input_pics, output_file, output_size=200, grid_size=None, height_field_size=1,
                 light_angle=60, w_g=1.5, w_s=0.001, radius=10, steps=1000):
        super().__init__(input_pics, output_file, output_size=output_size, grid_size=grid_size, height_field_size=height_field_size, 
        light_angle=light_angle, w_g=w_g, w_s=w_s, radius=radius, steps=steps)

        self.metrices = Metrices(grid_size=self.grid_size)

    @analyze    
    def make_step(self, row, col, delta):
        status, delta_obj = super().make_step(row, col, delta)
        self.metrices.update_pix(row, col, status)
        return status, delta_obj

    def step(self):
        delta = 0
        while 0 == delta:
            delta = np.random.randint(-5, 6)
        p = self.metrices.pix_score.reshape(-1) / self.metrices.total_score
        idx = np.random.choice(self.metrices.pix_score.size, 1, p=p)[0]
        row = idx // self.grid_size
        col = idx % self.grid_size
        return self.make_step(row, col, delta)