# ShadowPix

Implementation of **[SHADOWPIX: Multiple Images from Self Shadowing](https://www.cs.tau.ac.il/~amberman/shadowpixPaper.pdf)** paper.

Input (image)            |  Output (mesh)
:-------------------------:|:-------------------------:
![](orig.gif)  |  ![](model.gif)

## Local Method 
Example of local method usage -

**Simple run:**
```js
~/ShadowPix
❯ python local_method.py
Mesh saved into local_method.obj # <--- This file will contain your mesh
```

**Choose your own pictures and target file:**
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

The main idea is to reduce the number of failed attempts of changing the height of a pixel when we choose it randomly. PixModel gives scores to each pixel in the grid, representing how likely is it to be updated successfuly - the scoring is a weighted sum of failed/succcesful updates, and failed/succesful updates of neighbor pixels. (see code and paper for further explainations)

The user can control the hyperparameters of PixModel:
- with_bias (bool, True): run PixModel along with biased weights from objective function (equally weighted)
- min_score (float, 0.1): the minimum score a pixel has in the PixModel
- gain (float, 0.5): the score a pixel gains when it is succesfully updated
- punish (float, -0.15): the score a pixel loses when it fails to be updated
- neighbor_factor (float, 0.07): when a pixel updates its score, it updates its neighbor with `(gain|punish)*neighbor_factor` 
- neighbor_radius (int, 1): when a pixel updates its score, it updates its neighbors of radius neighbor_radius with `(gain|punish)*neighbor_factor` 

**Train the model:**
```js
~/ShadowPix master* ⇣
❯ python model/train.py
Strat training
0.0% success:0.0%,success_rand:0.0%, fail1:0.0%,fail2:0.0% obj_value:71206.07482022583
0.05% success:33.56643356643357%,success_rand:1.4985014985014986%, fail1:0.0%,fail2:66.33366633366633% obj_value:66327.78198100696
```

**Train the model using custom pics and targets:**
```js
~/Dev/yariv/ShadowPix master* ⇣ 9s
❯ python model/train.py -p pics/pic_1.jpg pics/pic_2.jpg pics/pic_3.jpg pics/pic_4.jpg -o amazing_mesh.obj 
Strat training
0.0% success:0.0%,success_rand:0.0%, fail1:0.0%,fail2:0.0% obj_value:99696.17530237656
0.05% success:53.54645354645355%,success_rand:2.6973026973026974%, fail1:0.0%,fail2:46.353646353646354% obj_value:90854.34548646204
```
**See full usage:**
```js
~/ShadowPix
❯ python model/train.py --help                                                                            
usage: train.py [-h] [-p [PICS [PICS ...]]] [-o OUTPUT] [--output-size OUTPUT_SIZE] [--wall-size WALL_SIZE]
                [--pixel-size PIXEL_SIZE] [-i ITERATIONS] [--height-field-size HEIGHT_FIELD_SIZE]
                [-l LIGHT_ANGLE] [-g GRADIENT_WEIGHT] [-s SMOOTH_WEIGHT] [-b] [--min-score MIN_SCORE]
                [--gain GAIN] [--punish PUNISH] [--neighbor-factor NEIGHBOR_FACTOR]
                [--neighbor-radius NEIGHBOR_RADIUS] [--log-path LOG_PATH] [-v]

ShadowPix global method with PixModel

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
  -b, --with-bias       Wether to include ussage of biased costs method
  --min-score MIN_SCORE
                        Minimum score in PixModel
  --gain GAIN           Incremental score in PixModel for successful updates
  --punish PUNISH       Decremental score in PixModel for failed updates
  --neighbor-factor NEIGHBOR_FACTOR
                        Reducing factor for update value for neighboring pixels
  --neighbor-radius NEIGHBOR_RADIUS
                        Neighboring pixel radius (how much steps to go further)
  --log-path LOG_PATH   Path to log PixModel statistics
  -v, --verbose-log     If set, all pixel data is logged in log_path
```


