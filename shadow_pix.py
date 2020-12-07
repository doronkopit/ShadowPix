from local_method import LocalMethod
import image_util

pics = ["pics/pic_a.jpg", "pics/pic_b.jpg", "pics/pic_c.jpg"]
output = 'local_output'
output_size = 200  # size in milemeters of output print
wall_size = 0.1  # thickness of walls
pixel_size = 2.5
grid_size = int(output_size / (wall_size + pixel_size))

square_imgs = [image_util.load_pic_to_square_np(pic, grid_size) for pic in pics]
for im in square_imgs:
    image_util.show_image(im)

local_m=LocalMethod(square_imgs, output,)
