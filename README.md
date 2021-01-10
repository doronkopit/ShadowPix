# ShadowPix

Implementation of **[SHADOWPIX: Multiple Images from Self Shadowing](https://www.cs.tau.ac.il/~amberman/shadowpixPaper.pdf)** paper.

Input (image)            |  Output (mesh)
:-------------------------:|:-------------------------:
![](orig.gif)  |  ![](model.gif)

## Local Method 
**Example of local method usage -**

**Can simply run it:**
```js
~/ShadowPix
❯ python local_method.py
Mesh saved into local_method.obj
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
  -c, --with-chamfers   Wether to use chamfers

``` 

## Global Method 
Example of global method usage, using your own pictures and parameters -
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


