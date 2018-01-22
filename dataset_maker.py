# -*- coding: utf-8 -*-
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
from ssd_utils import BBoxUtility
from ssd import SSD300 as SSD
import os


def run_camera(input_shape, model, save_path, frame_number):
    num_classes = 21
    conf_thresh = 0.4
    bbox_util = BBoxUtility(num_classes)

    class_colors = []
    for i in range(0,num_classes):
        hue = 255 * i / num_classes
        col = np.zeros((1, 1, 3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128  # Saturation
        col[0][0][2] = 255  # Value
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)

    vid = cv2.VideoCapture(0)

    # Compute aspect ratio of video
    vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # vidar = vidw / vidh
    samples = os.listdir(save_path)
    sample_count = len(samples)
    empty_count = 0
    image_stack = []
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
                empty_count += 1
                if empty_count == 4:
                    image_stack = []
                    empty_count = 0
            else:
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
                        cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax),
                                      class_colors[class_num], 2)
                        frame = orig_image
                        if len(image_stack) < frame_number:
                            image_stack.append(frame[ymin:ymax, xmin:xmax, :])
                        if len(image_stack) == frame_number:
                            image_stack.pop(0)
                            image_stack.append(frame[ymin:ymax, xmin:xmax, :])
        cv2.imshow("SSD result", orig_image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(5) & 0xFF == ord('s'):
            if len(image_stack) == frame_number:
                if not os.path.exists(save_path+str(sample_count+1)):
                    os.mkdir(save_path+str(sample_count+1))
                for pic in range(frame_number):
                    cv2.imwrite(save_path+str(sample_count+1)+'/' +
                                str(1000+pic)+'.jpg', image_stack[pic])
                    print('saving ' + save_path+str(sample_count+1)+'/' +
                                str(1000+pic)+'.jpg')
                image_stack = []
                empty_count = 0
                sample_count += 1


if __name__ == '__main__':
    action_class = 'stand/'
    root_path = 'images/'
    save_path = root_path+action_class
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_frames = 16
    input_shape = (300, 300, 3)
    ssd_model = SSD(input_shape, num_classes=21)
    ssd_model.load_weights('weights_SSD300.hdf5')
    run_camera(input_shape, ssd_model, save_path, save_frames)
