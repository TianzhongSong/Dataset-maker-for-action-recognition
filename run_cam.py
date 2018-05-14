# coding=utf8
import time

import cv2
import os

from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path


fps_time = 0


if __name__ == '__main__':
    # 类别以及要保存的视频段长度
    action = 'satnd'
    clip_length = 90
    root_path = '/home/dl1/datasets/actions/'
    if not os.path.exists(root_path + action):
        os.mkdir(root_path + action)
    if not os.path.exists(root_path + action + '/txt/'):
        os.mkdir(root_path + action + '/txt/')
        os.mkdir(root_path + action + '/imgs/')
    samples = len(os.listdir(root_path + action + '/txt/'))
    sample_count = 1000 if samples == 0 else 1000 + samples

    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
    cam = cv2.VideoCapture(0)
    ret_val, image = cam.read()
    joints = []
    joints_imgs = []
    while True:
        ret_val, image = cam.read()
        if ret_val:
            humans = e.inference(image)
            image, joint, *_, sk = TfPoseEstimator.get_humans(image, humans, imgcopy=False)
            if joint:
                if len(joints) < clip_length:
                    joints.append(joint[0])
                    joints_imgs.append(sk)
                else:
                    joints.pop(0)
                    joints_imgs.pop(0)
                    joints.append(joint[0])
                    joints_imgs.append(sk)
            cv2.putText(image,
                        "FPS: %.2f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            k = cv2.waitKey(5) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                if len(joints) == clip_length:
                    img_path = root_path + action + '/imgs/' + str(sample_count) + '/'
                    if not os.path.exists(img_path):
                        os.mkdir(img_path)
                    with open(root_path + action + '/txt/' + str(sample_count) + '.txt', 'w') as f:
                        for i in range(clip_length):
                            points = ""
                            for point in joints[i]:
                                points = points + str(point) + "_" + str(joints[i][point][0]) +\
                                         "_" + str(joints[i][point][1]) + " "
                            points = points[0:-1] + "\n"
                            f.write(points)
                            cv2.imwrite(img_path + str(1000 + i) + '.jpg', joints_imgs[i])
                        f.close()
                    sample_count += 1
                    joints = []
                    joints_imgs = []
    cam.release()
    cv2.destroyAllWindows()
