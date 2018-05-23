# Dataset-maker-for-action-recognition

----------------------Using openpose---------------------------------

## Attention：There can be only one person in the field of vision.

Get human joint information using [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), in this demo I use [TensorFlow implementation](https://github.com/ildoonet/tf-pose-estimation)

[This implementation](https://github.com/ildoonet/tf-pose-estimation) is trained with [COCO](http://mscoco.org/), there are 18 joints of a person.

    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/sksk.jpg)

## run

    ./pose/models/pretrained/mobilenet_v1_0.75_224_2017_06_14/download.sh
    
    python run_cam.py

Press 's' to save joint information and joint images during running, press 'q' to quit.

The default camera resolution is 640x480, the format of saved joint is t_x_y, where 't' indicates the number of joint, 'x' indicates the horizontal location of joint on the image, 'y' indicates the vertical location of joint on the image.

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/joints.png)

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/1000.jpg)

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/imgs.png)


----------------------Using SSD----------------------------------
## Attention：There can be only one person in the field of vision.

#Using [SSD](https://github.com/rykov8/ssd_keras) to detect person then save person crop image.

## run 
    python dataset_maker.py

Press 's' to save frames.

samples:

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/walk.jpg)
