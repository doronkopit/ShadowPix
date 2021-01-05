from scipy.optimize import differential_evolution as minimi
import cv2
import numpy as np
import image_util
import global_method

radius = 10
grid_size = 100
directions = 4
W_G = 1.5
W_S = 0.001
pics = ["pics/pic_a.jpg", "pics/pic_b.jpg", "pics/pic_c.jpg", "pics/pic_d.jpg"]
output = 'global_abcd.obj'
square_imgs = [image_util.load_pic_to_square_np(pic, 100) for pic in pics]
x = np.zeros(grid_size* grid_size)

glbm = global_method.GlobalMethod(square_imgs, output, output_size=100)


def calc_objective_val(h):
    L = cal_l_all(h)
    l_conv_p = lp_conv(L)
    l_conv_p_conv_g = grad_conv(l_conv_p)
    h_conv_g = grad_conv(h)
    parts = np.zeros(3)
    parts[0] = global_method.mse(l_conv_p, glbm.pics)
    parts[1] = W_G * global_method.mse(l_conv_p_conv_g, glbm.gradient_pass_filter_images)
    parts[2] = W_S * global_method.mse(h_conv_g, None)
    return parts.sum()


def cal_l_all(H):
    H=H.reshape([grid_size,grid_size])
    L = np.ones([directions, grid_size, grid_size])
    compare_idx_vector = np.arange(1, radius + 1)
    mat_select = np.arange(0, grid_size).reshape([grid_size, 1]) + compare_idx_vector
    for row in range(0, grid_size):
        new_l = L.copy()
        vector = None
        for d in range(directions):
            if d == 0:
                vector = H[row, :]
            elif d == 1:
                vector = H[row, ::-1]
            elif d == 2:
                vector = H[:, row]
            elif d == 3:
                vector = H[::-1, row]
            vect_w_radius = add_rad_2_vec(vector)
            comp_matrix = vect_w_radius[mat_select] - compare_idx_vector
            comp_matrix = comp_matrix.max(axis=1)
            l_update = np.clip((vector - comp_matrix), 0, 1)
            if d == 1 or d == 3:
                l_update = l_update[::-1]
            if d < 2:
                new_l[d, row, :] = l_update
            else:
                new_l[d, :, row] = l_update
        L = new_l
        return L

def add_rad_2_vec(vector):
        res = np.ones(grid_size + radius) * (-2000)
        res[:vector.shape[0]] = vector
        return res

def grad_conv(pics):
    if type(pics) != np.ndarray or len(pics.shape) > 2:
        g_x = np.array([cv2.Sobel(pic, cv2.CV_64F, 1, 0, ksize=5) for pic in pics])
        g_y = np.array([cv2.Sobel(pic, cv2.CV_64F, 0, 1, ksize=5) for pic in pics])
    else:
        g_x = cv2.Sobel(pics, cv2.CV_64F, 1, 0, ksize=5)
        g_y = cv2.Sobel(pics, cv2.CV_64F, 0, 1, ksize=5)

    return np.sqrt(np.power(g_x, 2) + np.power(g_y, 2))


def lp_conv(pics):
    if type(pics) != np.ndarray or len(pics.shape) > 2:
        return np.array([cv2.blur(pic, (5, 5)) for pic in pics])
    return cv2.blur(pics, (5, 5))
Nfeval=1
def callbackF(Xi,f=None,context=None):
    global Nfeval
    if f is None:
        f=calc_objective_val(Xi)
    if context is None:
        context=''
    print ('{0:4d}   {1: 3.6f}  {2:2d}'.format(Nfeval, f,context))
    Nfeval += 1
print("starting to minimize")
bnds=np.zeros([grid_size*grid_size,2])
bnds[:,1]=1000
#res = minimi(calc_objective_val,x, callback=callbackF,method='Powell')
#res = minimi(calc_objective_val,bnds, callback=callbackF,workers=-1)
res = minimi(calc_objective_val,bnds, disp=True,workers=-1)

glbm.h = res.x.reshape([grid_size,grid_size])
glbm.export_to_obj()
