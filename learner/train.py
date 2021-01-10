import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from learner.global_method_learning import GlobalMethodLearner
from learner.learning_utils import log_statistics
from util import image_util
import argparse


if __name__ == '__main__':
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    parser = argparse.ArgumentParser(description='ShadowPix global method')
    parser.add_argument('-p', '--pics', nargs='*',
                        default=["pics/pic_a.jpg",
                                 "pics/pic_b.jpg",
                                 "pics/pic_c.jpg",
                                 "pics/pic_d.jpg"],
                        help="List of strings representing grayscale images to use")
    parser.add_argument('-o', '--output',
                        default='global_method.obj',
                        type=str, help="Output filename for resulting .OBJ file")
    parser.add_argument('--output-size',
                        default=200, type=int, help="Output file size in mm")
    parser.add_argument('--wall-size',
                        default=0.25, type=float, help="Thickness of walls in output file")
    parser.add_argument('--pixel-size',
                        default=2.5, type=float, help="Pixel size of output file")
    parser.add_argument("-i", "--iterations",
                        default=2 * 10 ** 6, type=int, help="Number of iterations to perform (see paper)")
    parser.add_argument("--height-field-size",
                        default=1, type=int, help="Size of resulting heightfield")
    parser.add_argument("-l", "--light-angle",
                        default=60, type=int, help="Target theta angle of mesh")
    parser.add_argument("-g", "--gradient-weight",
                        default=1.5, type=float, help="Weight of gradient term in objective function (see paper)")
    parser.add_argument("-s", "--smooth-weight",
                        default=0.001, type=float, help="Weight of smooth term in objective function (see paper)")
    parser.add_argument('-b', '--with-bias',
                        default=True, action='store_true', help="Wether to include ussage of biased costs method")
    parser.add_argument("--min-score",
                        default=0.1, type=float, help="Minimum score in PixModel")
    parser.add_argument("--gain",
                        default=0.5, type=float, help="Incremental score in PixModel for successful updates")
    parser.add_argument("--punish",
                        default=-0.15, type=float, help="Decremental score in PixModel for failed updates")
    parser.add_argument("--neighbor-factor",
                        default=0.07, type=float, help="Reducing factor for update value for neighboring pixels")
    parser.add_argument("--neighbor-radius",
                        default=1, type=int, help="Neighboring pixel radius (how much steps to go further)")
    parser.add_argument('--log-path',
                        type=str, help="Path to log PixModel statistics")
    parser.add_argument('-v', '--verbose-log',
                        action='store_true', help="If set, all pixel data is logged in log_path")
    args = parser.parse_args()

    # Fetch params
    pics = args.pics
    output = args.output
    output_size = args.output_size
    wall_size = args.wall_size
    pixel_size = args.pixel_size
    grid_size = int(output_size / (wall_size + pixel_size))
    steps = args.iterations
    height_field_size = args.height_field_size
    light_angle = args.light_angle
    gradient_weight = args.gradient_weight
    smooth_weight = args.smooth_weight
    with_bias = args.with_bias

    # PixModel params
    min_score = args.min_score
    gain = args.gain
    punish = args.punish
    neighbor_factor = args.neighbor_factor
    neighbor_radius = args.neighbor_radius
    log_path = args.log_path
    verbose_logs = args.verbose_log

    res = 1
    square_imgs = [image_util.load_pic_to_square_np(pic, output_size // res) for pic in pics]

    global_m = GlobalMethodLearner(input_pics=square_imgs,
                                   output_file=output,
                                   output_size=output_size,
                                   steps=steps,
                                   height_field_size=1,
                                   light_angle=light_angle,
                                   weight_G=gradient_weight,
                                   weight_S=smooth_weight,
                                   with_bias=with_bias,
                                   min_score=min_score,
                                   gain=gain,
                                   punish=punish,
                                   neighbor_factor=neighbor_factor,
                                   neighbor_radius=neighbor_radius)
    print("Strat training")
    global_m.produce_pix()
    log_statistics(global_m.model, log_path=log_path, log_all=verbose_logs)
