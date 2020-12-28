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
        self.u = np.zeros([self.grid_size, self.grid_size+ 1])
        self.v = None
        self.r = None
        # cplus and cminus are relatively to r
        self.cplus = np.zeros([self.grid_size, self.grid_size])
        self.cminus = np.zeros([self.grid_size, self.grid_size])

        #self.mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, self.output_size, 0], [self.output_size, 0, 0],
        #                                      [self.output_size, self.output_size, 0]], faces=[[0, 1, 2], [1, 2, 3]])
        self.height = 0.0
        self.vertices = [None]
        self.faces = []

    def produce_pix(self):
        self.calc_constrains()
        self.export_constrains_to_mesh()
        self.save_mesh_to_output()

    def calc_constrains(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.u[j, i] = self.u[j, 0] + self.S * (
                    sum([self.pics[1][j, x] - self.pics[0][j, x] for x in range(1, j + 1)]))  # equation 1 in the paper
        self.u[0, :self.grid_size] += self.S * self.pics[0][:, 0]  # fix constraints of first column

        # eq3_constrains = self.S * (
        #       -self.pics[0][:, :self.grid_size - 1] + self.pics[0][:, 1:] - self.pics[2][:, 1:])
        eq3_constrains = np.zeros([self.grid_size-1, self.grid_size])

        # self.u[:, 0] -= min(0, np.min(self.u[:, 0]))  # setting the minimium of  u's first row to zero
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                eq3_constrains[j, i] = self.S * (-self.pics[0][j, i] + self.pics[0][i, j + 1] - self.pics[2][j+1, i])
                eq3_j_constrain = -self.u[j+1, i] + self.u[j, i] + eq3_constrains[j, i]
                self.u[j+1, i] += max(eq3_j_constrain, 0)
        self.r = self.u[:,:self.grid_size] - self.S * self.pics[0]
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
                if self.r[j, i] < self.r[j, i + 1]:
                    delta = min(self.r[j, i + 1] - self.r[j, i], self.u[j, i + 2] - self.r[j, i + 1])
                    if delta <= 0:
                        delta = 0
                    else:
                        self.change_delta_to_right(j, i + 1, -delta)
                    sum_of_d -= delta
                    self.cplus[j, i + 1] = delta
                    self.cminus[j, i] = 0
                elif self.r[j, i] > self.r[j, i + 1]:
                    delta = min(self.r[j, i] - self.r[j, i + 1], self.u[j, i] - self.r[j, i],
                                self.receiver_size - (self.cplus[j, i] / self.S))
                    if delta <= 0:
                        delta = 0
                    else:
                        self.change_delta_to_right(j, i + 1, delta)
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
        for i in range(self.grid_size + 1):
            for j in range(self.grid_size):
                self.create_wall_mesh(i,j, self.u[j, i])
                if i != self.grid_size:
                    self.create_receiver_mesh(i,j, self.r[j,i])
                    self.create_plus_chamfer(i,j, self.cplus[j,i])
                    self.create_minus_chamfer(i,j, self.cminus[j,i])
                    self.create_vwall_mesh(i,j, self.v[j,i])

    def save_mesh_to_output(self):
        print("ready to show")
        with open(self.output_path, 'w+') as f:
            for v in self.vertices:
                if v is None:
                    continue
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0], face[1], face[2]))

    def create_wall_mesh(self, i, j, param):
        # creates 5 parts of a wall block
        lwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                   [i * self.unit_size, (j + 1) * self.unit_size, param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(lwall)
        rwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size + self.wall_size, j * self.unit_size, 0],
                                                   [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size,
                                                    param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(rwall)
        upwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, (j + 1) * self.unit_size, 0],
                                                    [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size,
                                                     param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(upwall)
        dwnwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                     [i * self.unit_size + self.wall_size, j * self.unit_size,
                                                      param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(dwnwall)
        topwall = mesh_util.get_4_points_from_2_horiz([i * self.unit_size, j * self.unit_size, param],
                                                      [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size,
                                                       param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(topwall)

    def create_receiver_mesh(self, i, j, param):
        receiver = mesh_util.get_4_points_from_2_horiz([i * self.unit_size + self.wall_size, j * self.unit_size, param],
                                                       [(i + 1) * self.unit_size,
                                                        (j + 1) * self.unit_size - self.wall_size, param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(receiver)

    def create_plus_chamfer(self, i, j, param):
        chamferx_dist = param / self.S
        points = []
        points.append([(i + 1) * self.unit_size - chamferx_dist, j * self.unit_size + self.wall_size, self.r[j,i]])
        points.append([(i + 1) * self.unit_size, j * self.unit_size + self.wall_size, self.r[j,i] + param])
        points.append([(i + 1) * self.unit_size, (j + 1) * self.unit_size, self.r[j,i] + param])
        points.append([(i + 1) * self.unit_size - chamferx_dist, (j + 1) * self.unit_size, self.r[j,i]])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(points)

    def create_minus_chamfer(self, i, j, param):
        chamferx_dist = param / self.S
        points = []
        points.append(
            [i * self.unit_size + self.wall_size + chamferx_dist, j * self.unit_size + self.wall_size, self.r[j,i]])
        points.append([i * self.unit_size + self.wall_size, j * self.unit_size + self.wall_size, self.r[j,i] + param])
        points.append(
            [i * self.unit_size + self.wall_size, (j + 1) * self.unit_size, self.r[j,i] + param])
        points.append([i * self.unit_size + self.wall_size + chamferx_dist, (j + 1) * self.unit_size, self.r[j,i]])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(points)

    def create_vwall_mesh(self, i, j, param):
        # creates 5 parts of a wall block
        lwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                   [i * self.unit_size, j * self.unit_size + self.wall_size, param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(lwall)

        rwall = mesh_util.get_4_points_from_2_vert([(i + 1) * self.unit_size, j * self.unit_size, 0],
                                                   [(i + 1) * self.unit_size, j * self.unit_size + self.wall_size,
                                                    param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(rwall)

        upwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size + self.wall_size, 0],
                                                    [(i + 1) * self.unit_size, j * self.unit_size + self.wall_size,
                                                     param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(upwall)

        dwnwall = mesh_util.get_4_points_from_2_vert([i * self.unit_size, j * self.unit_size, 0],
                                                     [(i + 1) * self.unit_size, j * self.unit_size, param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(dwnwall)

        topwall = mesh_util.get_4_points_from_2_horiz([i * self.unit_size, j * self.unit_size, param],
                                                      [(i + 1) * self.unit_size, j * self.unit_size + self.wall_size,
                                                       param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(topwall)
