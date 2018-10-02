# DancingGaga
[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose
) implementation using darknet framework, originated from [openpose-darknet](https://github.com/lincolnhard/openpose-darknet)

# Result

![demo](https://raw.githubusercontent.com/jing-interactive/DancingGaga/master/doc/gaga1.gif)

# Steps to build from Visual Studio 2015

- First you need to build [lightnet](https://github.com/jing-vision/lightnet)
  - `git clone --recurse-submodules https://github.com/jing-vision/lightnet.git`
  - Follow [lightnet's building steps](https://github.com/jing-vision/lightnet#how-to-build-from-visual-studio-2015)
- Then you need to have [premake](https://premake.github.io/download.html) installed and execute `DancingGaga/gen-vs2015.bat` to generate `DancingGaga/vs2015` folder
- You can find `DancingGaga/vs2015/DancingGaga.sln`, you should be able to build it w/o errors. (If you are lucky like me.)


# Steps to run

- Download [weight file](https://drive.google.com/open?id=1BfY0Hx2d2nm3I4JFh0W1cK2aHD1FSGea) and copy it as `bin/openpose.weight`
  
- Usage
```
DancingGaga.exe -cfg=[openpose.cfg] -weights=[openpose.weight] media-source
```
e.g you can detect pose from a video
```
DancingGaga.exe pickme-101.mp4
```
Or from an image
```
DancingGaga.exe person.jpg
```
Or even from your default camera (index #0)
```
DancingGaga.exe 0
```
- Other network models
```
DancingGaga.exe -cfg=..\coco.cfg -weights=..\coco.weights person.jpg
DancingGaga.exe -cfg=..\mpi.cfg -weights=..\mpi.weights person.jpg
DancingGaga.exe -cfg=..\body_25.cfg -weights=..\body_25.weights person.jpg
```
# network layout

```
layer     filters    size              input                output
   0 conv     64  3 x 3 / 1   200 x 200 x   3   ->   200 x 200 x  64 0.138 BF
   1 conv     64  3 x 3 / 1   200 x 200 x  64   ->   200 x 200 x  64 2.949 BF
   2 max          2 x 2 / 2   200 x 200 x  64   ->   100 x 100 x  64 0.003 BF
   3 conv    128  3 x 3 / 1   100 x 100 x  64   ->   100 x 100 x 128 1.475 BF
   4 conv    128  3 x 3 / 1   100 x 100 x 128   ->   100 x 100 x 128 2.949 BF
   5 max          2 x 2 / 2   100 x 100 x 128   ->    50 x  50 x 128 0.001 BF
   6 conv    256  3 x 3 / 1    50 x  50 x 128   ->    50 x  50 x 256 1.475 BF
   7 conv    256  3 x 3 / 1    50 x  50 x 256   ->    50 x  50 x 256 2.949 BF
   8 conv    256  3 x 3 / 1    50 x  50 x 256   ->    50 x  50 x 256 2.949 BF
   9 conv    256  3 x 3 / 1    50 x  50 x 256   ->    50 x  50 x 256 2.949 BF
  10 max          2 x 2 / 2    50 x  50 x 256   ->    25 x  25 x 256 0.001 BF
  11 conv    512  3 x 3 / 1    25 x  25 x 256   ->    25 x  25 x 512 1.475 BF
  12 conv    512  3 x 3 / 1    25 x  25 x 512   ->    25 x  25 x 512 2.949 BF
  13 conv    256  3 x 3 / 1    25 x  25 x 512   ->    25 x  25 x 256 1.475 BF
  14 conv    128  3 x 3 / 1    25 x  25 x 256   ->    25 x  25 x 128 0.369 BF
  15 conv    128  3 x 3 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.184 BF
  16 conv    128  3 x 3 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.184 BF
  17 conv    128  3 x 3 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.184 BF
  18 conv    512  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 512 0.082 BF
  19 conv     38  1 x 1 / 1    25 x  25 x 512   ->    25 x  25 x  38 0.024 BF
  20 route  14
  21 conv    128  3 x 3 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.184 BF
  22 conv    128  3 x 3 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.184 BF
  23 conv    128  3 x 3 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.184 BF
  24 conv    512  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 512 0.082 BF
  25 conv     19  1 x 1 / 1    25 x  25 x 512   ->    25 x  25 x  19 0.012 BF
  26 route  19 25 14
  27 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  28 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  29 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  30 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  31 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  32 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  33 conv     38  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  38 0.006 BF
  34 route  26
  35 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  36 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  37 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  38 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  39 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  40 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  41 conv     19  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  19 0.003 BF
  42 route  33 41 14
  43 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  44 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  45 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  46 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  47 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  48 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  49 conv     38  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  38 0.006 BF
  50 route  42
  51 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  52 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  53 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  54 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  55 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  56 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  57 conv     19  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  19 0.003 BF
  58 route  49 57 14
  59 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  60 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  61 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  62 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  63 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  64 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  65 conv     38  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  38 0.006 BF
  66 route  58
  67 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  68 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  69 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  70 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  71 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  72 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  73 conv     19  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  19 0.003 BF
  74 route  65 73 14
  75 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  76 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  77 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  78 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  79 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  80 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  81 conv     38  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  38 0.006 BF
  82 route  74
  83 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  84 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  85 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  86 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  87 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  88 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  89 conv     19  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  19 0.003 BF
  90 route  81 89 14
  91 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
  92 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  93 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  94 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  95 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
  96 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
  97 conv     38  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  38 0.006 BF
  98 route  90
  99 conv    128  7 x 7 / 1    25 x  25 x 185   ->    25 x  25 x 128 1.450 BF
 100 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
 101 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
 102 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
 103 conv    128  7 x 7 / 1    25 x  25 x 128   ->    25 x  25 x 128 1.004 BF
 104 conv    128  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x 128 0.020 BF
 105 conv     19  1 x 1 / 1    25 x  25 x 128   ->    25 x  25 x  19 0.003 BF
 106 route  105 97
```

# Note

1. Darknet version openpose.cfg and openpose.weight are ported from COCO version 

  [pose_deploy_linevec.prototxt](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/tree/master/model/_trained_COCO) and [pose_iter_440000.caffemodel](  http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel).

2. You could change net input width, height in openpose.cfg.
