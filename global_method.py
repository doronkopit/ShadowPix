import numpy as np
import torch
import torch.nn as nn


class GlobalMethod:
    def __init__(self, input_pics, output_file, output_size=200, grid_size=None, height_field_size=0.5,
                 light_angle=60, w_g=0.1, w_s=0.005, radius=10, steps=1000):

        if len(input_pics) != 4:
            raise
        self.pics = [pic for pic in input_pics]  # inverting grayscale so black will be 1
        self.grad_conv, self.lp_conv = self.init_conv()
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
        self.radius_shadow_pos = np.arange(1, self.radius + 1)
        self.radius_shadow_neg = self.radius_shadow_pos[::-1]
        self.w_g = w_g
        self.w_s = w_s
        self.T = 0
        self.steps = steps
        self.L = np.ones([len(self.pics),grid_size, grid_size])
        self.obj_value = self.calc_objective_val(self.L)
        self.l_calculator = ShadowCalculatorL(self.radius, self.grid_size, len(self.pics)).calc_new_l

    def produce_pix(self):
        self.optimize()
        self.export_to_obj()

    def optimize(self):
        for i in range(self.steps):
            status = self.step()

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
            if self.delta_to_objective_enough(new_objective):
                self.L = new_l
                self.obj_value = new_objective
                return new_objective
            else:
                self.h[row, col] -= delta
                return -2

        else:
            self.h[row, col] -= delta
            return -1

    def calc_objective_val(self, L):
        l_conv_p = self.lp_conv(L)
        l_conv_p_conv_g = self.grad_conv(l_conv_p)
        h_conv_g = self.grad_conv
        parts = np.zeros(3)
        parts[0] = mse(l_conv_p, self.pics)
        parts[1] = self.w_g * mse(l_conv_p_conv_g, self.gradient_pass_filter_images)
        parts[2] = self.w_s * mse(h_conv_g)

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
                possible_shad = self.h[row, col + 1:col + self.radius + 1] + self.radius_shadow_pos
            elif d == 2 and row > self.radius:  # check down
                possible_shad = self.h[row - self.radius - 1:row, col] + self.radius_shadow_neg
            elif d == 3 and row < self.grid_size - self.radius - 1:  # check up
                possible_shad = self.h[row + 1:row + self.radius + 1, col] + self.radius_shadow_pos
            else:
                continue
            if possible_shad.max() < value_to_check:  # the shadow is casted more than the radius
                return False

        return True

    def delta_to_objective_enough(self, new_objective) -> bool:
        # todo implement
        return False

    def export_to_obj(self):
        # todo implement
        pass

    @staticmethod
    def init_conv():
        sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x.weight = nn.Parameter(torch.from_numpy(sobelx))
        conv_y.weight = nn.Parameter(torch.from_numpy(sobely))

        lp_filter = np.array([1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9])
        conv_lp = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_lp.weight = nn.Parameter(torch.from_numpy(lp_filter))

        def grad_conv(pics):
            g_x = conv_x(pics)
            g_y = conv_y(pics)
            return np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))

        def lp_conv(pics):
            return conv_lp(pics)

        return grad_conv, lp_conv


class ShadowCalculatorL:
    def __init__(self, radius, grid_size, directions):
        self.compare_idx_vector = np.arange(1, radius + 1)
        self.mat_select = np.arange(0, grid_size).reshape([grid_size, 1]) + self.compare_idx_vector
        self.num_of_directions = directions

    def calc_new_l(self, H, L, row, col):
        new_l = L.copy()
        for d in range(self.num_of_directions):
            if d==0:
                vector=H[row,:]
            elif d==1:
                vector=H[row,::-1]
            elif d==2:
                vector=H[:,col]
            elif d==3


        return 0


def mse(a, b):
    if b is None:
        b = np.zeros(a.shape)
    res = a - b
    res = np.sqrt(res ** 2)
    res = res.sum()
    return res
