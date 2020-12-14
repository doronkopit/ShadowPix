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
        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = int(output_size / self.unit_size)
        self.output_size = self.grid_size * self.unit_size + self.wall_size
        self.light_angle = light_angle
        self.S = 1 / np.tan(self.light_angle * (np.pi / 180))
        self.u = np.zeros([self.grid_size + 1, self.grid_size])
        self.v = None
        self.r = None
        # cplus and cminus are relatively to r
        self.cplus = np.zeros([self.grid_size, self.grid_size])
        self.cminus = np.zeros([self.grid_size, self.grid_size])

        self.mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, self.output_size, 0], [self.output_size, 0, 0],
                                              [self.output_size, self.output_size, 0]], faces=[[0, 1, 2], [1, 2, 3]])
        self.height = 0.0

    def produce_pix(self):
        self.calc_constrains()
        self.export_constrains_to_mesh()
        self.save_mesh_to_output()

    def calc_constrains(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.u[i, j] = self.u[0, j] + self.S * (
                    sum([self.pics[1][x, j] - self.pics[0][x, j] for x in range(1, j + 1)])) # equation 1 in the paper
        self.u[0, :] += self.S * self.pics[0][0, :]  # fix constraints of first column

        eq3_constrains = self.S * (
                -self.pics[0][:, :self.grid_size - 1] + self.pics[0][:, 1:] - self.pics[2][:, 1:])
        # self.u[:, 0] -= min(0, np.min(self.u[:, 0]))  # setting the minimium of  u's first row to zero
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                eq3_j_constrain = -self.u[i, j + 1] + self.u[i, j] + eq3_constrains[i, j]
                self.u[i, j + 1] += max(np.max(eq3_j_constrain), 0)
        self.r = self.u[:self.grid_size, :] - self.S * self.pics[0]
        # todo check if the order is right
        self.calc_chamfers()
        self.v = self.r + self.S * self.pics[2]
        min_height = min(np.min(self.u), np.min(self.r), np.min(self.v))
        self.u -= min_height
        self.r -= min_height
        self.v -= min_height
        self.height = max(np.max(self.u), np.max(self.r), np.max(self.v))
        print(f"height of model {self.height}")
        print("finished constrains")

    def calc_chamfers(self):
        sum_of_d = 0
        for j in range(self.grid_size):
            for i in range(self.grid_size - 1):
                if self.r[i, j] < self.r[i + 1, j]:
                    delta = min(self.r[i + 1, j] - self.r[i, j], self.u[i + 2, j] - self.r[i + 1, j])
                    if delta <= 0:
                        delta = 0
                    else:
                        self.change_delta_to_right(i + 1, j, -delta)
                    sum_of_d -= delta
                    self.cplus[i + 1, j] = delta
                    self.cminus[i, j] = 0
                elif self.r[i, j] > self.r[i + 1, j]:
                    delta = min(self.r[i, j] - self.r[i + 1, j], self.u[i, j] - self.r[i, j],
                                self.receiver_size - (self.cplus[i, j] / self.S))
                    if delta <= 0:
                        delta = 0
                    else:
                        self.change_delta_to_right(i + 1, j, delta)
                    sum_of_d += delta
                    self.change_delta_to_right(i + 1, j, delta)
                    self.cminus[i, j] = delta
                    self.cplus[i + 1, j] = 0
        print(sum_of_d)

    def change_delta_to_right(self, start_i, j, factor):
        if factor < 0:
            self.u[start_i, j] += factor
        for i in range(start_i, self.grid_size):
            self.u[i + 1, j] += factor
            self.r[i, j] += factor

    def export_constrains_to_mesh(self):
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size):
                cell_mesh = self.create_wall_mesh(i, j, self.u[i, j])
                if i != self.grid_size:
                    cell_mesh += self.create_receiver_mesh(i, j, self.r[i, j])
                    cell_mesh += self.create_plus_chamfer(i, j, self.cplus[i, j])
                    cell_mesh += self.create_minus_chamfer(i, j, self.cminus[i, j])
                    cell_mesh += self.create_vwall_mesh(i, j, self.v[i, j])
                self.mesh += cell_mesh

    def save_mesh_to_output(self):
        print("ready to show")
        self.mesh.export(self.output_path)

    def create_wall_mesh(self, i, j, param):
        # creates 5 parts of a wall block
        lwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                   [i * self.unit_size, (j + 1) * self.unit_size, param])
        wall_mesh = mesh_util.get_rectangle_mesh(lwall)
        rwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size + self.wall_size, j * self.unit_size, 0],
                                                   [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size,
                                                    param])
        wall_mesh += mesh_util.get_rectangle_mesh(rwall)
        upwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, (j + 1) * self.unit_size, 0],
                                                    [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size,
                                                     param])
        wall_mesh += mesh_util.get_rectangle_mesh(upwall)
        dwnwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                     [i * self.unit_size + self.wall_size, j * self.unit_size,
                                                      param])
        wall_mesh += mesh_util.get_rectangle_mesh(dwnwall)
        topwall = mesh_util.get_4_points_from_2_horiz([i * self.unit_size, j * self.unit_size, param],
                                                      [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size,
                                                       param])
        wall_mesh += mesh_util.get_rectangle_mesh(topwall)

        return wall_mesh

    def create_receiver_mesh(self, i, j, param):
        receiver = mesh_util.get_4_points_from_2_horiz([i * self.unit_size + self.wall_size, j * self.unit_size, param],
                                                       [(i + 1) * self.unit_size,
                                                        (j + 1) * self.unit_size - self.wall_size, param])
        return mesh_util.get_rectangle_mesh(receiver)

    def create_plus_chamfer(self, i, j, param):
        chamferx_dist = param / self.S
        points = []
        points.append([(i + 1) * self.unit_size - chamferx_dist, j * self.unit_size + self.wall_size, self.r[i, j]])
        points.append([(i + 1) * self.unit_size, j * self.unit_size + self.wall_size, self.r[i, j] + param])
        points.append([(i + 1) * self.unit_size, (j + 1) * self.unit_size, self.r[i, j] + param])
        points.append([(i + 1) * self.unit_size - chamferx_dist, (j + 1) * self.unit_size, self.r[i, j]])
        return mesh_util.get_rectangle_mesh(points)

    def create_minus_chamfer(self, i, j, param):
        chamferx_dist = param / self.S
        points = []
        points.append(
            [i * self.unit_size + self.wall_size + chamferx_dist, j * self.unit_size + self.wall_size, self.r[i, j]])
        points.append([i * self.unit_size + self.wall_size, j * self.unit_size + self.wall_size, self.r[i, j] + param])
        points.append(
            [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size, self.r[i, j] + param])
        points.append([i * self.unit_size + self.wall_size + chamferx_dist, (j + 1) * self.unit_size, self.r[i, j]])
        return mesh_util.get_rectangle_mesh(points)

    def create_vwall_mesh(self, i, j, param):
        # creates 5 parts of a wall block
        lwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                   [i * self.unit_size, j * self.unit_size + self.wall_size, param])
        wall_mesh = mesh_util.get_rectangle_mesh(lwall)
        rwall = mesh_util.get_4_points_from_2_vert([(i + 1) * self.unit_size, j * self.unit_size, 0],
                                                   [(i + 1) * self.unit_size, j * self.unit_size + self.wall_size,
                                                    param])
        wall_mesh += mesh_util.get_rectangle_mesh(rwall)
        upwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size + self.wall_size, 0],
                                                    [(i + 1) * self.unit_size, j * self.unit_size + self.wall_size,
                                                     param])
        wall_mesh += mesh_util.get_rectangle_mesh(upwall)
        dwnwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                     [(i + 1) * self.unit_size, j * self.unit_size, param])
        wall_mesh += mesh_util.get_rectangle_mesh(dwnwall)
        topwall = mesh_util.get_4_points_from_2_horiz([i * self.unit_size, j * self.unit_size, param],
                                                      [(i + 1) * self.unit_size, j * self.unit_size + self.wall_size,
                                                       param])
        wall_mesh += mesh_util.get_rectangle_mesh(topwall)
        return wall_mesh
