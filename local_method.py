import numpy as np
import image_util


class LocalMethod:
    def __init__(self,input_pics, output_file, output_size=200, grid_size=None, wall_size=0.25, unit_size=2.5,
                 light_angle=60):
        if len(input_pics) != 3:
            raise
        self.pics = [1 - pic for pic in input_pics]  # inverting grayscale so black will be 1
        self.output_path = output_file
        self.output_size = output_size
        self.wall_size = wall_size
        self.unit_size = unit_size
        if not grid_size:
            self.grid_size = int(self.output_size / (self.wall_size + self.pixel_size))
        self.light_angle = light_angle
        self.S = 1 / np.tan(self.light_angle)
        self.u = np.zeroes([self.grid_size, self.grid_size + 1])
        self.v = None
        self.r = None
        self.cplus = None
        self.cminus = None
        self.mesh = None

    def produce_pix(self):
        self.calc_constrains()
        self.exp_constrains_to_mesh()
        self.save_mesh_to_output()
