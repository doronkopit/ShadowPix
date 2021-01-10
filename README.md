# ShadowPix

Implementation of **[SHADOWPIX: Multiple Images from Self Shadowing](https://www.cs.tau.ac.il/~amberman/shadowpixPaper.pdf)** paper.

Input (image)            |  Output (mesh)
:-------------------------:|:-------------------------:
![](orig.gif)  |  ![](model.gif)

## Local Jethod 
To use **ShadowPix** visualization -
```js
~/ShadowPix
❯ python local_method.py
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
