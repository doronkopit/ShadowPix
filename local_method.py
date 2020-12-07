import numpy as np
import image_util


class LocalMethod:
    def __init__(self, input_pics, output_file, output_size=200, grid_size=None, wall_size=0.25, unit_size=2.5,
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
        self.S = 1 / np.tan(self.light_angle * (np.pi / 180))
        self.u = np.zeroes([self.grid_size, self.grid_size + 1])
        self.v = None
        self.r = None
        # cplus and cminus are relatively to r
        self.cplus = None
        self.cminus = None

        self.mesh = None

    def produce_pix(self):
        self.calc_constrains()
        self.export_constrains_to_mesh()
        self.save_mesh_to_output()

    def calc_constrains(self):
        self.u[:, 0] += self.S * self.pics[0][:, 0]  # fix constraints of first column
        for i in range(self.grid_size):
            self.u[:, i + 1] = self.u[:, i] + self.S * (
                    self.pics[1][:, i] - self.pics[0][:, i])  # equation 1 in the paper

        eq3_constrains = self.S * (-self.pics[0][:self.grid_size - 1, :] + self.pics[0][1:, :] - self.pics[2][1:, :])
        self.u[0, :] -= min(0, np.min(self.u[0, :]))  # setting the minimium of  u's first row to zero
        for j in range(self.grid_size - 1):
            eq3_j_constrain = -self.u[j + 1, :-1] + self.u[j, :-1] + eq3_constrains[j, :]
            self.u[j + 1, :] += max(np.max(eq3_j_constrain), 0)
        self.r = self.u[:, : self.grid_size] - self.S * self.pics[0]
        self.v = self.r + self.S * self.pics[2]
        self.calc_chamfers()

    def calc_chamfers(self):
        for j in range(self.grid_size):
            for i in range(self.grid_size - 1):
                if self.r[j, i] < self.r[j, i + 1]:
                    delta = min(self.r[j, i + 1] - self.r[j, i], self.u[j, i + 2] - self.r[j, i + 1])
                    self.change_delta_to_right(j, i + 1, -delta)
                    self.cplus[j, i + 1] = delta
                    self.cminus[j, i] = 0
                elif self.r[j, i] > self.r[j, i + 1]:
                    delta = min(self.r[j, i] - self.r[j, i + 1], self.u[j, i] - self.r[j, i], self.S - self.cplus[i, j])
                    self.change_delta_to_right(j, i + 1, delta)
                    self.cminus[j, i] = delta
                    self.cplus[j, i + 1] = 0

    def export_constrains_to_mesh(self):
        pass

    def save_mesh_to_output(self):
        pass

    def change_delta_to_right(self, j, start_i, factor):
        if factor < 0:
            self.u[j, start_i] += factor
        for i in range(start_i, self.grid_size):
            self.u[j, i + 1] += factor
            self.r[j, i] += factor
