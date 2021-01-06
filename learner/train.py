import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from learner.global_method_learning import GlobalMethodLearner
from learner.analyze_tools import print_stats
import image_util
import cProfile, pstats, io
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
pics = ["pics/pic_a.jpg", "pics/pic_b.jpg", "pics/pic_c.jpg", "pics/pic_d.jpg"]
output = 'global_init_test2.obj'
output_size = 200  # size in milemeters of output print
wall_size = 0.25  # thickness of walls
pixel_size = 2.5
grid_size = int(output_size / (wall_size + pixel_size))

local = False
res=1

square_imgs = [image_util.load_pic_to_square_np(pic, output_size // res) for pic in pics]
global_m = GlobalMethodLearner(square_imgs, output, output_size,steps=2*10**3,height_field_size=1)
print("starting optimize")
global_m.produce_pix()
print_stats(global_m.metrices)

print("finish")
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
