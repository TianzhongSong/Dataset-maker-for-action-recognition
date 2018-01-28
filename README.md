# Dataset-maker-for-action-recognition
用于制作人体行为识别数据库的程序

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
