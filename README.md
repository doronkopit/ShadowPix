# ShadowPix

Implementation of **[SHADOWPIX: Multiple Images from Self Shadowing](https://www.cs.tau.ac.il/~amberman/shadowpixPaper.pdf)** paper.

Input (image)            |  Output (mesh)
:-------------------------:|:-------------------------:
![](orig.gif)  |  ![](model.gif)

## Local Method 
Example of local method usage -

**Can simply run it:**
```js
~/ShadowPix
❯ python local_method.py
Mesh saved into local_method.obj # <--- This file will contain your mesh
```

**Or you can choose your own pictures and target file:**
```js
~/ShadowPix
❯ python local_method.py -p pics/pic_1.jpg pics/pic_2.jpg pics/pic_3.jpg -o awesome_mesh.obj
Mesh saved into awesome_mesh.obj
```

**See full usage:**
```js
~/ShadowPix
❯ python local_method.py --help

usage: local_method.py [-h] [-p [PICS [PICS ...]]] [-o OUTPUT] [--output-size OUTPUT_SIZE]
                       [--wall-size WALL_SIZE] [--pixel-size PIXEL_SIZE] [-c]

ShadowPix local method

optional arguments:
  -h, --help            show this help message and exit
  -p [PICS [PICS ...]], --pics [PICS [PICS ...]]
                        List of strings representing grayscale images to use
  -o OUTPUT, --output OUTPUT
                        Output filename for resulting .OBJ file
  --output-size OUTPUT_SIZE
                        Output file size in mm
  --wall-size WALL_SIZE
                        Thickness of walls in output file
  --pixel-size PIXEL_SIZE
                        Pixel size of output file
  -c, --with-chamfers   Wether to use chamfers

``` 

## Global Method 
Example of global method usage -

**Simple run:**
```js
~/ShadowPix master* ⇣
❯ python global_method.py 
Starting optimization of global method
0.0% success:0.0%,success_rand:0.0%, fail1:0.0%,fail2:0.0% obj_value:71206.07482022583
0.05% success:41.05894105894106%,success_rand:2.197802197802198%, fail1:0.0%,fail2:58.841158841158844% obj_value:65412.1092668208 
# Note that convergence might take some time
```

**Use custom pictures and targets:**
```js
~/ShadowPix
❯ python global_method.py -p pics/pic_1.jpg pics/pic_2.jpg pics/pic_3.jpg pics/pic_4.jpg -o amazing_mesh.obj
Starting optimization of global method
0.0% success:0.0%,success_rand:0.0%, fail1:0.0%,fail2:0.0% obj_value:99696.17530237656
0.05% success:55.24475524475525%,success_rand:1.3986013986013985%, fail1:0.0%,fail2:44.655344655344656% obj_value:90098.86939958937
```

**See full usage:**
```js
~/ShadowPix
❯ python global_method.py

usage: global_method.py [-h] [-p [PICS [PICS ...]]] [-o OUTPUT] [--output-size OUTPUT_SIZE]
                        [--wall-size WALL_SIZE] [--pixel-size PIXEL_SIZE] [-i ITERATIONS]
                        [--height-field-size HEIGHT_FIELD_SIZE] [-l LIGHT_ANGLE] [-g GRADIENT_WEIGHT]
                        [-s SMOOTH_WEIGHT] [-b]

ShadowPix global method

optional arguments:
  -h, --help            show this help message and exit
  -p [PICS [PICS ...]], --pics [PICS [PICS ...]]
                        List of strings representing grayscale images to use
  -o OUTPUT, --output OUTPUT
                        Output filename for resulting .OBJ file
  --output-size OUTPUT_SIZE
                        Output file size in mm
  --wall-size WALL_SIZE
                        Thickness of walls in output file
  --pixel-size PIXEL_SIZE
                        Pixel size of output file
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations to perform (see paper)
  --height-field-size HEIGHT_FIELD_SIZE
                        Size of resulting heightfield
  -l LIGHT_ANGLE, --light-angle LIGHT_ANGLE
                        Target theta angle of mesh
  -g GRADIENT_WEIGHT, --gradient-weight GRADIENT_WEIGHT
                        Weight of gradient term in objective function (see paper)
  -s SMOOTH_WEIGHT, --smooth-weight SMOOTH_WEIGHT
                        Weight of smooth term in objective function (see paper)
  -b, --biased-costs    Wether to use biased costs method
``` 

## PixModel (learning model) 

PixModel attempts to add a learning layer to the global method, using a scoring model for each pixel in the target heightfield (see paper). 

