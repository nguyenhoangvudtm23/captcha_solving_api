import tensorflow as tf
import keras
from glob import glob
import os
from yolov3.utils import *
from yolov3.configs import *
import cv2

model_directory = 'yolov4-tiny_pvi.h5'
Yolo = keras.models.load_model(model_directory)


def resize_image(image, target_size=(93, 375, 3)):
    ih, iw, _ = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = np.array(cv2.resize(image, (nw, nh)), dtype=np.uint8)

    image_paded = np.zeros((ih, iw, _), dtype=np.uint8)
    image_paded.fill(255)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized

    return image_paded


def preprocess(image_path):
    or_image = cv2.imread(image_path)
    # or_image = cv2.resize(or_image, (375, 93))
    # or_image = cv2.fastNlMeansDenoisingColored(or_image, None, 5, 7, 3, 21)
    image = cv2.cvtColor(or_image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # image = cv2.bitwise_not(image)

    thresh = cv2.bitwise_not(image)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        # if the contour has no other contours inside of it
        # if the size of the contour is less than a threshold (noise)
        if cv2.contourArea(cnt) < 80:
            # Fill the holes in the original image
            cv2.drawContours(thresh, [cnt], 0, 0, -1)
    # return cv2.bitwise_not(thresh)
    thresh = np.array(thresh / 255, dtype=np.uint8)
    new_thresh = np.stack((thresh, thresh, thresh), axis=-1)
    new_image = np.multiply(or_image, new_thresh)
    return new_image


def load_yolov4_tiny_model(model_path=model_directory):
    model = keras.models.load_model(model_path)
    model.summary()
    return model


def detect_image(image_path, input_size=416, CLASSES=TRAIN_CLASSES, score_threshold=0.55, iou_threshold=0.45):
    original_image = cv2.imread(image_path)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = preprocess(image_path)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = Yolo.predict(image_data)
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='soft-nms')

    # get string characters
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    # print(NUM_CLASS)
    # print(num_classes)
    new_bboxes = [box for box in bboxes if box[4] >= score_threshold]
    new_bboxes.sort(key=lambda x: (x[0] + x[2])/2)
    str = ''
    for b in new_bboxes:
        str += "{}".format(NUM_CLASS[int(b[5])])
    return str


if __name__ == '__main__':
    # image = cv2.imread('../test.png')
    # new_image = preprocess('../test.png')
    # cv2.imshow('image', new_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    directory = r'D:\Study\PythonPrj\pythonProject\Captcha_project\Detect\IMAGES\Test_PVI'
    out_dir = r'D:\Study\PythonPrj\pythonProject\Captcha_project\Detect\IMAGES\Test_PVI'
    correct = 0
    total = 0
    for file in os.listdir(directory):
        if not file.endswith('.png'):
            continue
        total += 1
        image_dir = os.path.join(directory, file)
        result = detect_image(image_dir)
        if result == file.split('.')[0]:
            correct += 1

        # os.rename(image_dir, os.path.join(out_dir, f'{result}.png'))

    print(f'accuracy: {correct}/{total}')
