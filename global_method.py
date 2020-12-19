import numpy as np


class GlobalMethod:
    def __init__(self, input_pics, output_file, output_size=200, grid_size=None, wall_size=0.25, receiver_size=2.5,
                 light_angle=60,w_g=0.1,w_s=0.005,radius=10):

        if len(input_pics) != 4:
            raise
        self.pics = [1 - pic for pic in input_pics]  # inverting grayscale so black will be 1
        self.output_path = output_file