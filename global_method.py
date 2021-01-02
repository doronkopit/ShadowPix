import cv2
import numpy as np
import torch
import torch.nn as nn
import mesh_util
import scipy.signal as signal


class GlobalMethod:
    def __init__(self, input_pics, output_file, output_size=200, grid_size=None, height_field_size=1,
                 light_angle=60, w_g=1.5, w_s=0.001, radius=10, steps=1000):

        if len(input_pics) != 4:
            raise
        self.pics = [pic for pic in input_pics]  # inverting grayscale so black will be 1
        self.grad_conv, self.lp_conv = self.init_conv()
        self.conv_meth = 'scipy'
        self.gradient_pass_filter_images = self.grad_conv(self.lp_conv(self.pics))
        self.output_path = output_file
        self.height_field_size = height_field_size
        self.output_size = output_size
        if not grid_size:
            self.grid_size = int(output_size // self.height_field_size)
        self.h = np.zeros([self.grid_size, self.grid_size])
        self.light_angle = light_angle
        self.S = 1 / np.tan(self.light_angle * (np.pi / 180))
        self.radius = radius
        self.radius_shadow_pos = np.arange(1, self.radius + 2)
        self.radius_shadow_neg = self.radius_shadow_pos[::-1]
        self.w_g = w_g
        self.w_s = w_s
        self.T = 1
        self.steps = steps
        self.alpha = self.T / self.steps
        self.L = np.ones([len(self.pics), self.grid_size, self.grid_size])
        self.obj_value = self.calc_objective_val(self.L)
        self.l_calculator = ShadowCalculatorL(self.radius, self.grid_size, len(self.pics)).calc_new_l
        self.vertices = [None]
        self.faces = []
        points = [[0, 0, 0], [0, self.output_size, 0], [self.output_size, self.output_size, 0],
                  [self.output_size, 0, 0]]
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(points)

    def produce_pix(self):
        self.optimize()
        self.export_to_obj()

    def optimize(self):
        fails1 = 0
        fails2 = 0
        success = 0
        success_rand = 0
        while success < self.steps:
            if success % 100 == 0:
                print(f'{success/self.steps}% success:{success},success_rand:{success_rand}, fail1:{fails1},fail2:{fails2} obj_value:{self.obj_value}')
            status, delta_obj = self.step()
            if status > 0:
                if delta_obj == 1:
                    success_rand += 1
                success += 1
            elif status == -1:
                fails1 += 1
            else:
                fails2 += 1
        floor = self.h.min()
        self.h -= floor

    def step(self):
        delta = 0
        while 0 == delta:
            delta = np.random.randint(-5, 6)
        row = np.random.randint(0, self.grid_size)
        col = np.random.randint(0, self.grid_size)
        self.h[row, col] += delta
        if self.valid_step(row, col, delta):
            new_l = self.l_calculator(self.h, self.L, row, col)
            new_objective = self.calc_objective_val(new_l)
            delta_obj = self.delta_to_objective_enough(new_objective)
            if delta_obj > 0:
                self.L = new_l
                self.T -= self.alpha
                self.obj_value = new_objective
                return new_objective, delta_obj
            else:
                self.h[row, col] -= delta
                return -2, None

        else:
            self.h[row, col] -= delta
            return -1, None

    def calc_objective_val(self, L):
        l_conv_p = self.lp_conv(L)
        l_conv_p_conv_g = self.grad_conv(l_conv_p)
        h_conv_g = self.grad_conv(self.h)
        parts = np.zeros(3)
        parts[0] = mse(l_conv_p, self.pics)
        parts[1] = self.w_g * mse(l_conv_p_conv_g, self.gradient_pass_filter_images)
        parts[2] = self.w_s * mse(h_conv_g, None)
        return parts.sum()

    def valid_step(self, row, col, delta) -> bool:
        # the function checks that the change doesnt effect a range
        # bigger than the radius
        if delta > 0:
            return self.valid_step_pos(row, col)
        else:
            return self.valid_step_neg(row, col)

    def valid_step_neg(self, row, col) -> bool:
        for d in range(len(self.pics)):
            if d == 0 and col > self.radius:  # check left
                value_to_check = self.h[row, col - self.radius - 1]
                possible_shad = self.h[row, col - self.radius:col + 1] + self.radius_shadow_pos
            elif d == 1 and col < self.grid_size - self.radius - 1:  # check right
                value_to_check = self.h[row, col + self.radius + 1]
                possible_shad = self.h[row, col:col + self.radius + 1] + self.radius_shadow_neg
            elif d == 2 and row > self.radius:  # check down
                value_to_check = self.h[row - self.radius - 1, col]
                possible_shad = self.h[row - self.radius:row + 1, col] + self.radius_shadow_pos
            elif d == 3 and row < self.grid_size - self.radius - 1:  # check up
                value_to_check = self.h[row + self.radius + 1, col]
                possible_shad = self.h[row:row + self.radius + 1, col] + self.radius_shadow_neg
            else:
                continue
            if possible_shad.max() < value_to_check:  # the shadow is casted more than the radius
                return False
        return True

    def valid_step_pos(self, row, col) -> bool:
        value_to_check = self.h[row, col]
        for d in range(len(self.pics)):
            if d == 0 and col > self.radius:  # check left
                possible_shad = self.h[row, col - self.radius - 1:col] + self.radius_shadow_neg
            elif d == 1 and col < self.grid_size - self.radius - 1:  # check right
                possible_shad = self.h[row, col + 1:col + self.radius + 2] + self.radius_shadow_pos
            elif d == 2 and row > self.radius:  # check down
                possible_shad = self.h[row - self.radius - 1:row, col] + self.radius_shadow_neg
            elif d == 3 and row < self.grid_size - self.radius - 1:  # check up
                possible_shad = self.h[row + 1:row + self.radius + 2, col] + self.radius_shadow_pos
            else:
                continue
            if possible_shad.max() < value_to_check:  # the shadow is casted more than the radius
                return False

        return True

    def delta_to_objective_enough(self, new_objective):
        obj_delta = self.obj_value - new_objective
        if obj_delta > 0:
            return obj_delta
        else:
            if (np.random.random() < np.e ** (obj_delta / self.T)):
                return 1
            else:
                return -2

    def export_to_obj(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.h[i, j] != 0:
                    self.create_h_mesh(i, j, self.h[i, j])
        self.save_mesh_to_output()

    def save_mesh_to_output(self):
        print("ready to show")
        with open(self.output_path, 'w+') as f:
            for v in self.vertices:
                if v is None:
                    continue
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for face in self.faces:
                f.write("f %d %d %d\n" % (face[0], face[1], face[2]))

    @staticmethod
    def init_conv():
        sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

        # sobelx_tor = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # sobely_tor = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # conv_x.weight = nn.Parameter(sobelx_tor.unsqueeze(0).unsqueeze(0))
        # conv_y.weight = nn.Parameter(sobely_tor.unsqueeze(0).unsqueeze(0))

        lp_filter = np.ones([3, 3], np.float32) / 9

        # lp_filter_tor = torch.Tensor([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
        # conv_lp = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # conv_lp.weight = nn.Parameter(lp_filter_tor.unsqueeze(0).unsqueeze(0))

        def grad_conv(pics):
            if type(pics) != np.ndarray or len(pics.shape) > 2:
                # g_x = np.array([signal.fftconvolve(pic, sobelx, mode='same') for pic in pics])
                # g_y = np.array([signal.fftconvolve(pic, sobely, mode='same') for pic in pics])

                # g_x = np.array([conv_x(torch.from_numpy(pic)) for pic in pics])
                # g_y = np.array([conv_y(torch.from_numpy(pic)) for pic in pics])

                g_x = np.array([cv2.Sobel(pic, cv2.CV_64F, 1, 0, ksize=3) for pic in pics])
                g_y = np.array([cv2.Sobel(pic, cv2.CV_64F, 0, 1, ksize=3) for pic in pics])

            else:

                # g_x = signal.fftconvolve(pics, sobelx, mode='same')
                # g_y = signal.fftconvolve(pics, sobely, mode='same')
                # g_x = conv_x(torch.from_numpy(pics))
                # g_y = conv_y(torch.from_numpy(pics))
                g_x = cv2.Sobel(pics, cv2.CV_64F, 1, 0, ksize=3)
                g_y = cv2.Sobel(pics, cv2.CV_64F, 0, 1, ksize=3)

            return np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))

        def lp_conv(pics):
            # return conv_lp(pics)
            if type(pics) != np.ndarray or len(pics.shape) > 2:
                # return np.array([signal.fftconvolve(pic, lp_filter, mode='same') for pic in pics])
                return np.array([cv2.blur(pic, (3, 3)) for pic in pics])
                # pics=[torch.from_numpy(pic).unsqueeze(0).unsqueeze(0).float() for pic in pics]
                # return np.array([conv_lp(pic) for pic in pics])
            # return signal.fftconvolve(pics, lp_filter, mode='same')
            return cv2.blur(pics, (3, 3))

            # return conv_lp(pics)

        return grad_conv, lp_conv

    def create_h_mesh(self, i, j, param):
        # creates 5 parts of a wall block
        lwall = mesh_util.get_4_points_from_2_vert([i * self.height_field_size, j * self.height_field_size, 0],
                                                   [i * self.height_field_size, (j + 1) * self.height_field_size,
                                                    param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(lwall)
        rwall = mesh_util.get_4_points_from_2_vert([(i + 1) * self.height_field_size, j * self.height_field_size, 0],
                                                   [(i + 1) * self.height_field_size, (j + 1) * self.height_field_size,
                                                    param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(rwall)
        upwall = mesh_util.get_4_points_from_2_vert([i * self.height_field_size, (j + 1) * self.height_field_size, 0],
                                                    [(i + 1) * self.height_field_size, (j + 1) * self.height_field_size,
                                                     param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(upwall)
        dwnwall = mesh_util.get_4_points_from_2_vert([i * self.height_field_size, j * self.height_field_size, 0],
                                                     [(i + 1) * self.height_field_size, j * self.height_field_size,
                                                      param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(dwnwall)
        topwall = mesh_util.get_4_points_from_2_horiz([i * self.height_field_size, j * self.height_field_size, param],
                                                      [(i + 1) * self.height_field_size,
                                                       (j + 1) * self.height_field_size,
                                                       param])
        verts = len(self.vertices)
        self.faces.extend([[verts, verts + 1, verts + 2], [verts, verts + 2, verts + 3]])
        self.vertices.extend(topwall)


class ShadowCalculatorL:
    def __init__(self, radius, grid_size, directions):
        self.grid_size = grid_size
        self.radius = radius
        self.compare_idx_vector = np.arange(1, radius + 1)
        self.mat_select = np.arange(0, grid_size).reshape([grid_size, 1]) + self.compare_idx_vector
        self.num_of_directions = directions

    def calc_new_l(self, H, L, row, col):
        new_l = L.copy()
        vector = None
        for d in range(self.num_of_directions):
            if d == 0:
                vector = H[row, :]
            elif d == 1:
                vector = H[row, ::-1]
            elif d == 2:
                vector = H[:, col]
            elif d == 3:
                vector = H[::-1, col]
            vect_w_radius = self.add_rad_2_vec(vector)
            comp_matrix = vect_w_radius[self.mat_select] - self.compare_idx_vector
            comp_matrix = comp_matrix.max(axis=1)
            l_update = np.clip((vector - comp_matrix), 0, 1)
            if d == 1 or d == 3:
                l_update = l_update[::-1]
            if d < 2:
                new_l[d, row, :] = l_update
            else:
                new_l[d, :, col] = l_update
        return new_l

    def add_rad_2_vec(self, vector):
        res = np.ones(self.grid_size + self.radius) * (-2000)
        res[:vector.shape[0]] = vector
        return res


def mse(a, b):
    if b is None:
        b = np.zeros(a.shape)
    res = a - b
    res = (res ** 2)
    res = (res.sum()) #/ a.size
    return res
    #return np.linalg.norm(a-b)
