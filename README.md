# Dataset-maker-for-action-recognition

用于制作人体行为识别数据库的程序

----------------------Update openpose---------------------------------

采用[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)获取人体关节点信息，并将骨骼图保存下来。我使用的是这个[openpose加速版](https://github.com/ildoonet/tf-pose-estimation)

run:

    ./pose/models/pretrained/mobilenet_v1_0.75_224_2017_06_14/download.sh
    
    python run_cam.py

运行过程中， 按‘s’键保存信息，按‘q’键退出。

保存的关节点信息：共有18个关节点信息，使用[run_cam.py](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/run_cam.py) 保存每一帧各个关节点的序号及其坐标，以及每一帧对应的骨骼图。

结果如下：

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/joints.png)

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/1000.jpg)

![](https://github.com/TianzhongSong/Dataset-maker-for-action-recognition/blob/master/imgs/imgs.png)

----------------------Prevous Version----------------------------------

#使用[SSD（含权重文件）](https://github.com/rykov8/ssd_keras)检测人体

## requirements
keras2.10+

tensorflow1.4+

opencv3.2+

## 说明
运行主程序：dataset_maker.py

1、将变量action_class设置为待采类别；

2、变量save_frames为一个样本的帧长度，也就是说每一个样本是由连续save_frames帧的图片所组成，
每一个样本都单独保存在一个文件夹中；

3、在采集过程中在摄像头窗口按‘s’键即保存当前图片。

知乎文章：[一个制作带bbox的行为识别数据库的程序](https://zhuanlan.zhihu.com/p/33365628)
