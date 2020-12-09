from typing import List, Any, Union

import numpy as np
import trimesh
import mesh_util


# import image_util


class LocalMethod:
    def __init__(self, input_pics, output_file, output_size=200, grid_size=None, wall_size=0.25, receiver_size=2.5,
                 light_angle=60):
        if len(input_pics) != 3:
            raise
        self.pics = [1 - pic for pic in input_pics]  # inverting grayscale so black will be 1
        self.output_path = output_file

        self.wall_size = wall_size
        self.receiver_size = receiver_size
        self.unit_size = self.receiver_size + self.wall_size
        if not grid_size:
            self.grid_size = int(output_size / self.unit_size)
        self.output_size = self.grid_size * self.unit_size + self.wall_size
        self.light_angle = light_angle
        self.S = 1 / np.tan(self.light_angle * (np.pi / 180))  # *self.receiver_size
        self.u = np.zeros([self.grid_size, self.grid_size + 1])
        self.v = None
        self.r = None
        # cplus and cminus are relatively to r
        self.cplus = np.zeros([self.grid_size, self.grid_size])
        self.cminus = np.zeros([self.grid_size, self.grid_size])

        self.mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, self.output_size, 0], [self.output_size, 0, 0],
                                              [self.output_size, self.output_size, 0]], faces=[[0, 1, 2], [1, 2, 3]])

    def produce_pix(self):
        self.calc_constrains()
        self.export_constrains_to_mesh()
        self.save_mesh_to_output()

    def calc_constrains(self):
        for i in range(self.grid_size):
            self.u[:, i + 1] = self.u[:, i] + self.S * (
                    self.pics[1][:, i] - self.pics[0][:, i])  # equation 1 in the paper
        self.u[:, 0] += self.S * self.pics[0][:, 0]  # fix constraints of first column

        eq3_constrains = self.S * (-self.pics[0][:self.grid_size - 1, :] + self.pics[0][1:, :] - self.pics[2][1:, :])
        self.u[0, :] -= min(0, np.min(self.u[0, :]))  # setting the minimium of  u's first row to zero
        for j in range(self.grid_size - 1):
            eq3_j_constrain = -self.u[j + 1, :-1] + self.u[j, :-1] + eq3_constrains[j, :]
            self.u[j + 1, :] += max(np.max(eq3_j_constrain), 0)
        self.r = self.u[:, : self.grid_size] - self.S * self.pics[0]
        # todo check if the order is right
        self.calc_chamfers()
        self.v = self.r + self.S * self.pics[2]
        print("finished constrains")

    def calc_chamfers(self):
        sum_of_d = 0
        for j in range(self.grid_size):
            for i in range(self.grid_size - 1):
                if self.r[j, i] < self.r[j, i + 1]:
                    delta = min(self.r[j, i + 1] - self.r[j, i], self.u[j, i + 2] - self.r[j, i + 1])
                    sum_of_d -= delta
                    self.change_delta_to_right(j, i + 1, -delta)
                    self.cplus[j, i + 1] = delta
                    self.cminus[j, i] = 0
                elif self.r[j, i] > self.r[j, i + 1]:
                    delta = min(self.r[j, i] - self.r[j, i + 1], self.u[j, i] - self.r[j, i], self.S - self.cplus[j, i])
                    sum_of_d += delta
                    self.change_delta_to_right(j, i + 1, delta)
                    self.cminus[j, i] = delta
                    self.cplus[j, i + 1] = 0
        print(sum_of_d)

    def change_delta_to_right(self, j, start_i, factor):
        if factor < 0:
            self.u[j, start_i] += factor
        for i in range(start_i, self.grid_size):
            self.u[j, i + 1] += factor
            self.r[j, i] += factor

    def export_constrains_to_mesh(self):
        for j in range(self.grid_size):
            for i in range(self.grid_size + 1):
                cell_mesh = self.create_wall_mesh(j, i, self.u[j, i])
                if i != self.grid_size:
                    cell_mesh += self.create_receiver_mesh(j, i, self.r[j, i])
                    cell_mesh += self.create_plus_chamfer(j, i, self.cplus[j, i])
                    cell_mesh += self.create_minus_chamfer(j, i, self.cminus[j, i])
                    cell_mesh += self.create_vwall_mesh(j, i, self.v[j, i])
                self.mesh += cell_mesh

    def save_mesh_to_output(self):
        print("ready to show")
        self.mesh.export(self.output_path)

    def create_wall_mesh(self, j, i, param):
        # creates 5 parts of a wall block
        lwall = mesh_util.get_4_points_from_2_vert([j * self.unit_size, i * self.unit_size, 0],
                                                   [(j + 1) * self.unit_size, i * self.unit_size, param])
        wall_mesh = mesh_util.get_rectangle_mesh(lwall)
        rwall = mesh_util.get_4_points_from_2_vert([j * self.unit_size, i * self.unit_size + self.wall_size, 0],
                                                   [(j + 1) * self.unit_size, i * self.unit_size + self.wall_size,
                                                    param])
        wall_mesh += mesh_util.get_rectangle_mesh(rwall)
        upwall = mesh_util.get_4_points_from_2_vert([(j + 1) * self.unit_size, i * self.unit_size, 0],
                                                    [(j + 1) * self.unit_size, i * self.unit_size + self.wall_size,
                                                     param])
        wall_mesh += mesh_util.get_rectangle_mesh(upwall)
        dwnwall = mesh_util.get_4_points_from_2_vert([j * self.unit_size, i * self.unit_size, 0],
                                                     [j * self.unit_size, i * self.unit_size + self.wall_size,
                                                      param])
        wall_mesh += mesh_util.get_rectangle_mesh(dwnwall)
        topwall = mesh_util.get_4_points_from_2_horiz([j * self.unit_size, i * self.unit_size, param],
                                                      [(j + 1) * self.unit_size, i * self.unit_size + self.wall_size,
                                                       param])
        wall_mesh += mesh_util.get_rectangle_mesh(topwall)

        return wall_mesh

    def create_receiver_mesh(self, j, i, param):
        receiver = mesh_util.get_4_points_from_2_horiz([j * self.unit_size, i * self.unit_size, param],
                                                       [(j + 1) * self.unit_size, (i + 1) * self.unit_size, param])
        return mesh_util.get_rectangle_mesh(receiver)

    def create_plus_chamfer(self, j, i, param):
        chamferx_dist = param / self.S
        points = []
        points.append([j * self.unit_size + self.wall_size, (i + 1) * self.unit_size - chamferx_dist, self.r[j, i]])
        points.append([j * self.unit_size + self.wall_size, (i + 1) * self.unit_size, self.r[j, i] + param])
        points.append([(j + 1) * self.unit_size, (i + 1) * self.unit_size, self.r[j, i] + param])
        points.append([(j + 1) * self.unit_size, (i + 1) * self.unit_size - chamferx_dist, self.r[j, i]])
        return mesh_util.get_rectangle_mesh(points)

    def create_minus_chamfer(self, j, i, param):
        chamferx_dist = param / self.S
        points = []
        points.append(
            [j * self.unit_size + self.wall_size, i * self.unit_size + self.wall_size + chamferx_dist, self.r[j, i]])
        points.append([j * self.unit_size + self.wall_size, i * self.unit_size + self.wall_size, self.r[j, i] + param])
        points.append([(j + 1) * self.unit_size, i * self.unit_size + self.wall_size, self.r[j, i] + param])
        points.append([(j + 1) * self.unit_size, i * self.unit_size + self.wall_size + chamferx_dist, self.r[j, i]])
        return mesh_util.get_rectangle_mesh(points)

    def create_vwall_mesh(self, j, i, param):
        # creates 5 parts of a wall block
        lwall = mesh_util.get_4_points_from_2_vert([j * self.unit_size, i * self.unit_size, 0],
                                                   [j * self.unit_size + self.wall_size, i * self.unit_size, param])
        wall_mesh = mesh_util.get_rectangle_mesh(lwall)
        rwall = mesh_util.get_4_points_from_2_vert([j * self.unit_size, (i + 1) * self.unit_size, 0],
                                                   [j * self.unit_size + self.wall_size, (i + 1) * self.unit_size,
                                                    param])
        wall_mesh += mesh_util.get_rectangle_mesh(rwall)
        upwall = mesh_util.get_4_points_from_2_vert([j * self.unit_size + self.wall_size, i * self.unit_size, 0],
                                                    [j * self.unit_size + self.wall_size, (i + 1) * self.unit_size,
                                                     param])
        wall_mesh += mesh_util.get_rectangle_mesh(upwall)
        dwnwall = mesh_util.get_4_points_from_2_vert([j * self.unit_size, i * self.unit_size, 0],
                                                     [j * self.unit_size, (i + 1) * self.unit_size, param])
        wall_mesh += mesh_util.get_rectangle_mesh(dwnwall)
        topwall = mesh_util.get_4_points_from_2_horiz([j * self.unit_size, i * self.unit_size, param],
                                                      [j * self.unit_size + self.wall_size, (i + 1) * self.unit_size,
                                                       param])
        wall_mesh += mesh_util.get_rectangle_mesh(topwall)
        return wall_mesh
