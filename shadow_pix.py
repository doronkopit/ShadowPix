from local_method import LocalMethod
from global_method import GlobalMethod
import image_util
import cProfile, pstats, io
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
pics = ["pics/pic_1.jpg", "pics/pic_2.jpg", "pics/pic_3.jpg", "pics/pic_4.jpg"]
output = 'global_output_123.obj'
output_size = 200  # size in milemeters of output print
wall_size = 0.25  # thickness of walls
pixel_size = 2.5
grid_size = int(output_size / (wall_size + pixel_size))

square_imgs = [image_util.load_pic_to_square_np(pic, 200) for pic in pics]
# for im in square_imgs:
#     image_util.show_image(im)
local = False
if local:
    local_m = LocalMethod(square_imgs[:3], output, output_size, grid_size=grid_size, wall_size=wall_size,
                          receiver_size=pixel_size)
    local_m.produce_pix()
else:
    global_m = GlobalMethod(square_imgs, output, output_size,steps=1000000)
    global_m.produce_pix()

print("finish")
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
