# -*- coding: utf-8 -*-
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
from ssd_utils import BBoxUtility
from ssd import SSD300 as SSD
import os
from time import sleep
import copy


def run_camera(input_shape, model, root_path, action_class, frame_number):
    num_classes = 21
    conf_thresh = 0.6
    bbox_util = BBoxUtility(num_classes)

    class_colors = []
    for i in range(0, num_classes):
        hue = 255 * i / num_classes
        col = np.zeros((1, 1, 3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128  # Saturation
        col[0][0][2] = 255  # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)

    vid = cv2.VideoCapture(0)
    sleep(2)
    # Compute aspect ratio of video
    vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # vidar = vidw / vidh
    crop_path = root_path + 'crop/' + action_class
    origin_path = root_path + 'origin/' + action_class
    mask_path = root_path + 'mask/' + action_class
    samples = os.listdir(origin_path)
    sample_count = len(samples)
    while True:
        retval, orig_image = vid.read()
        if not retval:
            print("Done!")
            return None

        im_size = (input_shape[0], input_shape[1])
        resized = cv2.resize(orig_image, im_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        inputs = [image.img_to_array(rgb)]
        tmp_inp = np.array(inputs)
        x = preprocess_input(tmp_inp)

        y = model.predict(x)

        results = bbox_util.detection_out(y)
        if len(results) > 0 and len(results[0]) > 0:
            det_label = results[0][:, 0]
            det_conf = results[0][:, 1]
            det_xmin = results[0][:, 2]
            det_ymin = results[0][:, 3]
            det_xmax = results[0][:, 4]
            det_ymax = results[0][:, 5]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            if 15 not in top_label_indices:
                detected = False
            else:
                detected = True
                for i in range(top_conf.shape[0]):
                    xmin = int(round((top_xmin[i] * vidw) * 0.9))
                    ymin = int(round((top_ymin[i] * vidh) * 0.9))
                    xmax = int(round((top_xmax[i] * vidw) * 1.1)) if int(
                        round((top_xmax[i] * vidw) * 1.1)) <= vidw else int(
                        round(top_xmax[i] * vidw))
                    ymax = int(round((top_ymax[i] * vidh) * 1.1)) if int(
                        round((top_ymax[i] * vidh) * 1.1)) <= vidh else int(
                        round(top_ymax[i] * vidh))

                    # save frames
                    class_num = int(top_label_indices[i])
                    if class_num == 15:
                        frame = copy.deepcopy(orig_image)
                        cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax),
                                      class_colors[class_num], 2)
                        curl = np.zeros_like(frame, dtype='uint8')
                        curl[ymin:ymax, xmin:xmax, :] = frame[ymin:ymax, xmin:xmax, :]
                        crop = cv2.resize(frame[ymin:ymax, xmin:xmax, :], (64, 96))
                        curl = cv2.resize(curl, (160, 120))
                        frame = cv2.resize(frame, (160, 120))
        else:
            detected = False

        cv2.imshow("SSD result", orig_image)
        if cv2.waitKey(5) & 0xFF == ord('s') and detected:
            sample_count += 1
            cv2.imwrite(crop_path + '/' + str(sample_count + 10000) + '.jpg', crop)
            print('saving ' + crop_path + '/' + str(sample_count + 10000) + '.jpg')
            cv2.imwrite(origin_path + '/' + str(sample_count + 10000) + '.jpg', frame)
            print('saving ' + origin_path + '/' + str(sample_count + 10000) + '.jpg')
            cv2.imwrite(mask_path + '/' + str(sample_count + 10000) + '.jpg', curl)
            print('saving ' + mask_path + '/' + str(sample_count + 10000) + '.jpg')


if __name__ == '__main__':
    pose_class = 'sit/'
    root_path = '/home/dl1/datasets/pose/'
    if not os.path.exists(root_path + 'crop/'):
        os.mkdir(root_path + 'crop/')
    if not os.path.exists(root_path + 'origin/'):
        os.mkdir(root_path + 'origin/')
    if not os.path.exists(root_path + 'mask/'):
        os.mkdir(root_path + 'mask/')

    save_path = root_path + pose_class
    if not os.path.exists(root_path + 'crop/' + pose_class):
        os.mkdir(root_path + 'crop/' + pose_class)
    if not os.path.exists(root_path + 'origin/' + pose_class):
        os.mkdir(root_path + 'origin/' + pose_class)
    if not os.path.exists(root_path + 'mask/' + pose_class):
        os.mkdir(root_path + 'mask/' + pose_class)

    save_frames = 30
    input_shape = (300, 300, 3)
    ssd_model = SSD(input_shape, num_classes=21)
    ssd_model.load_weights('weights_SSD300.hdf5')
    run_camera(input_shape, ssd_model, root_path, pose_class, save_frames)
