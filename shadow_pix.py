from local_method import LocalMethod
from global_method import GlobalMethod
from util import image_util
import cProfile, pstats, io
from pstats import SortKey
import argparse


if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    parser = argparse.ArgumentParser(description='ShadowPix quick demo')
    parser.add_argument('-g',
                        action='store_true', help="Wether to run global method demo (local as default)")
    args = parser.parse_args()

    pics = ["pics/pic_a.jpg", "pics/pic_b.jpg", "pics/pic_c.jpg", "pics/pic_d.jpg"]
    output_size = 200  # size in milemeters of output print
    wall_size = 0.25  # thickness of walls
    pixel_size = 2.5
    grid_size = int(output_size / (wall_size + pixel_size))

    local = not args.g
    output = ""
    if local:
        print("Starting local method demo")
        output = 'demo_local.obj'

        square_imgs = [image_util.load_pic_to_square_np(pic, grid_size) for pic in pics]
        local_m = LocalMethod(square_imgs[:3], output, output_size, grid_size=grid_size, wall_size=wall_size,
                              receiver_size=pixel_size)
        local_m.produce_pix()
    else:
        print("Starting global method demo")
        output = 'demo_global.obj'

        res=1
        square_imgs = [image_util.load_pic_to_square_np(pic, output_size // res) for pic in pics]
        global_m = GlobalMethod(square_imgs, output, output_size,steps=2*10**6,height_field_size=1)
        print("Starting global method optimization")
        global_m.produce_pix()

    print(f'Finished demo, check {output} file')
