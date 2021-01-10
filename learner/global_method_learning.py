from global_method import GlobalMethod
from learner.model import analyze, PixModel
import numpy as np


class GlobalMethodLearner(GlobalMethod):
    def __init__(self, input_pics, output_file, output_size=200, grid_size=None, height_field_size=1,
                 light_angle=60, weight_G=1.5, weight_S=0.001, radius=10, steps=1000, with_bias=True,
                 min_score=0.1, gain=0.5, punish=-0.15, neighbor_factor=0.07, neighbor_radius=1):
        super().__init__(input_pics, output_file, output_size=output_size, grid_size=grid_size, height_field_size=height_field_size,
                         light_angle=light_angle, weight_G=weight_G, weight_S=weight_S, radius=radius, steps=steps)

        self.model = PixModel(grid_size=self.grid_size,
                              min_score=min_score,
                              gain=gain,
                              punish=punish,
                              neighbor_factor=neighbor_factor,
                              radius=neighbor_radius)
        self.with_bias = with_bias

    @analyze    
    def make_step(self, row, col, delta):
        status, delta_obj = super().make_step(row, col, delta)
        self.model.update_pix(row, col, status)
        return status, delta_obj

    def step(self):
        delta = 0
        while 0 == delta:
            delta = np.random.randint(-5, 6)
        p = self.model.pix_score.reshape(-1) / self.model.total_score
        if self.with_bias:
            p = (p + self.idx_cost) / 2
        idx = np.random.choice(self.model.pix_score.size, 1, p=p)[0]
        row = idx // self.grid_size
        col = idx % self.grid_size
        return self.make_step(row, col, delta)