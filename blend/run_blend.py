import os
import sys
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import glob
import argparse
import imageio
from PIL import Image

project_root = 'C:\\Users\\Doron Kopit\\PycharmProjects\\ShadowPix'
parser = argparse.ArgumentParser()
parser.add_argument('--blender_path', type=str, default='/args/blender/blender', help='')
parser.add_argument('--script_path', type=str, default=project_root + '/visualize/render.py', help='')
parser.add_argument('--blend_path', type=str, default=project_root + '/visualize/stage2.blend', help='')
parser.add_argument('--input_folder', type=str, default=project_root + '/models/mies.obj', help='')
parser.add_argument('--output_folder', type=str, default=project_root + '/renders', help='')
parser.add_argument('--gif_frames', type=int, default=0, help='number of gif frames')
parser.add_argument('--duration', type=int, default=4, help='duration of gif in seconds')
args = parser.parse_args()


def render(input_obj_dir, output_folder):
    print('Generating rendering commands...')
    if os.path.isfile(input_obj_dir):
        fullfile = input_obj_dir
    else:
        return
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    file_id, file_extension = os.path.splitext(os.path.basename(fullfile))
    output_file = os.path.join(output_folder, file_id + '.png')
    command = ['%s %s --background --python %s -- %s %s %d > /dev/null 2>&1' % (
    args.blender_path, args.blend_path, args.script_path, fullfile, output_file, args.gif_frames)]
    print(command[0])
    print('Rendering')
    pool = Pool(1)
    for idx, return_code in enumerate(pool.imap(partial(call, shell=True), command)):
        if return_code != 0:
            print('Rendering command (\"%s\") failed' % (command[idx]))
    if args.gif_frames > 1:
        create_gif_local(file_id)
    print('done!')


def alpha_to_color(file_name, color=(255, 255, 255)):
    png = Image.open(file_name)
    if len(png.getpixel((0, 0))) == 4:
        png.load()
        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
        background.save(file_name)


def create_gif_local(base_name):
    images = []
    durations = []
    for i in range(args.gif_frames):
        file_name = '%s/%s_%d.png' % (args.output_folder, base_name, i)
        # alpha_to_color(file_name)
        if i == 0 or i == args.gif_frames - 1 or i == args.gif_frames / 2 - 1:
            durations.append(0.5)
        else:
            durations.append(args.duration / args.gif_frames)
        images.append(imageio.imread(file_name))
    for i in range(args.gif_frames):
        file_name = '%s/%s_%d.png' % (args.output_folder, base_name, args.gif_frames - i - 1)
        if i == args.gif_frames / 2:
            durations.append(0.3)
        else:
            durations.append(args.duration / args.gif_frames)
        # alpha_to_color(file_name)
        images.append(imageio.imread(file_name))
    imageio.mimsave('%s/%s_anim.gif' % (args.output_folder, base_name), images, duration=durations)


def create_gif_global(base_name):
    images = []
    durations = []
    for i in range(args.gif_frames):
        file_name = '%s/%s_%d.png' % (args.output_folder, base_name, i)
        alpha_to_color(file_name)
        if i % (args.gif_frames / 4) == 0:
            durations.append(2)
        else:
            durations.append(0.4)
        images.append(imageio.imread(file_name))
    imageio.mimsave('%s/%s_anim.gif' % (args.output_folder, base_name), images, duration=durations)


if __name__ == "__main__":
    args.gif_frames = 40
    create_gif_local('faces')
