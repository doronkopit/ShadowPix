import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from learner.global_method_learning import GlobalMethodLearner
from learner.learning_utils import print_stats
import image_util

if __name__ == '__main__':
    pics = ["pics/pic_a.jpg", "pics/pic_b.jpg", "pics/pic_c.jpg", "pics/pic_d.jpg"]
    output = 'global_init_test2.obj'
    output_size = 200  # size in milemeters of output print
    wall_size = 0.25  # thickness of walls
    pixel_size = 2.5
    grid_size = int(output_size / (wall_size + pixel_size))
    res=1

    square_imgs = [image_util.load_pic_to_square_np(pic, output_size // res) for pic in pics]
    global_m = GlobalMethodLearner(square_imgs, output, output_size,steps=2*10**4,height_field_size=1)
    print("Strat training")
    global_m.produce_pix()
    print_stats(global_m.model)
