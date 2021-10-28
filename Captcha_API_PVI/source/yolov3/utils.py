import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *


def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def image_preprocess1(image, target_size):
    ih, iw, _ = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = np.array(cv2.resize(image, (nw, nh)), dtype=np.uint8)

    image_paded = np.zeros((ih, iw, _), dtype=np.uint8)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized

    return image_paded


def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h,  w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def prepare_image1(image, bboxes):
    # bboxes: (xmin, ymin, xmax, ymax, score, class)
    img_h, img_w, _ = image.shape
    xmin, ymin = 10000, 10000
    xmax, ymax = 0, 0
    mask_image = np.zeros((img_h, img_w, _))
    if len(bboxes) == 0:
        return mask_image
    if len(bboxes) > 2:
        n = len(bboxes)
        # sort follow by score decreasing
        for i in range(n):
            for j in range(0, n-i-1):
                if bboxes[j][4] < bboxes[j+1][4]:
                    bboxes[j], bboxes[j+1] = bboxes[j+1], bboxes[j]

        bboxes = bboxes[:2]

    for bbox in bboxes:
        coor = np.array(bbox[:4], dtype=np.int32)
        x1, y1 = coor[0] - coor[2]//2, coor[1] - coor[3]//2
        x2, y2 = coor[0] + coor[2]//2, coor[1] + coor[3]//2
        # get small bb
        bw, bh = (x2 - x1), (y2 - y1)
        x1 = max(0, x1 - int(0.1 * bw))
        y1 = max(0, y1 - int(0.1 * bh))
        x2 = min(img_w - 1, x2 + int(0.1 * bw))
        y2 = min(img_h - 1, y2 + int(0.1 * bh))
        xmin = min(xmin, x1)
        ymin = min(ymin, y1)
        xmax = max(xmax, x2)
        ymax = max(ymax, y2)
        mask_image[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]
    new_image = mask_image[ymin:ymax, xmin:xmax, :]
    return new_image


def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors=''):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    lst = []
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        if score < 0.6:
            continue
        lst.append(bbox)
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1:
            bbox_thick = 1
        fontScale = 0.5 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            # license_string = recognize_characters(plate_img)
            # NUM_CLASS[class_ind]
            label = "{}".format(NUM_CLASS[class_ind])

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick) # cal and return the size of box that contains the specified text.
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    lst.sort(key=lambda x: (x[0] + x[2])/2)
    str = ''
    for b in lst:
        str += "{}".format(NUM_CLASS[int(b[5])])
    return image, str


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'): # non maximum suppression
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0 
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


# function for crop_bb
def prepare_image(image, bboxes):
    #bboxes: (xmin, ymin, xmax, ymax, score, class)
    img_h, img_w, _ = image.shape
    xmin, ymin = 10000, 10000
    xmax, ymax = 0, 0
    for bbox in bboxes:
        coor = np.array(bbox[:4], dtype=np.int32)
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        xmin = min(xmin, x1)
        ymin = min(ymin, y1)
        xmax = max(xmax, x2)
        ymax = max(ymax, y2)

    bw, bh = (xmax - xmin), (ymax - ymin)
    xmin = max(0, xmin - int(0.1 * bw))
    ymin = max(0, ymin - int(0.1 * bh))
    xmax = min(img_w-1, xmax + int(0.1 * bw))
    ymax = min(img_h-1, ymax + int(0.1 * bh))
    # image[:ymin, :, :] = np.zeros((ymin, img_w, _), dtype=int)
    # image[ymax+1:, :, :] = np.zeros((img_h-ymax-1, img_w, _),dtype=int)
    # image[:, :xmin, :] = np.zeros((img_h, xmin, _), dtype=int)
    # image[:, xmax+1:img_w, :] = np.zeros((img_h, img_w-xmax-1, _), dtype=int)
    new_image = image[ymin:ymax, xmin:xmax, :]
    return new_image


# crop only bounding box function
def crop_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.6,
               iou_threshold=0.7, resize=False, target_size=(224, 224, 3)):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='soft-nms')
    new_image = prepare_image(original_image, bboxes)
    if resize:
        new_image = image_preprocess1(new_image, target_size)
    if output_path != '':
        try:
            cv2.imwrite(output_path, new_image)
        except ValueError:
            pass


# function for crop_bb
def prepare_image1(image, bboxes):
    # bboxes: (xmin, ymin, xmax, ymax, score, class)
    img_h, img_w, _ = image.shape
    xmin, ymin = 10000, 10000
    xmax, ymax = 0, 0
    mask_image = np.zeros((img_h, img_w, _))
    if len(bboxes) == 0:
        return mask_image
    if len(bboxes) > 2:
        n = len(bboxes)
        # sort follow by score decreasing
        for i in range(n):
            for j in range(0, n-i-1):
                if bboxes[j][4] < bboxes[j+1][4]:
                    bboxes[j], bboxes[j+1] = bboxes[j+1], bboxes[j]

        bboxes = bboxes[:2]

    for bbox in bboxes:
        coor = np.array(bbox[:4], dtype=np.int32)
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        # get small bb
        bw, bh = (x2 - x1), (y2 - y1)
        x1 = max(0, x1 - int(0.1 * bw))
        y1 = max(0, y1 - int(0.1 * bh))
        x2 = min(img_w - 1, x2 + int(0.1 * bw))
        y2 = min(img_h - 1, y2 + int(0.1 * bh))
        xmin = min(xmin, x1)
        ymin = min(ymin, y1)
        xmax = max(xmax, x2)
        ymax = max(ymax, y2)
        mask_image[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]

    bw, bh = (xmax - xmin), (ymax - ymin)
    # xmin = max(0, xmin - int(0.1 * bw))
    # ymin = max(0, ymin - int(0.1 * bh))
    # xmax = min(img_w-1, xmax + int(0.1 * bw))
    # ymax = min(img_h-1, ymax + int(0.1 * bh))
    # image[:ymin, :, :] = np.zeros((ymin, img_w, _), dtype=int)
    # image[ymax+1:, :, :] = np.zeros((img_h-ymax-1, img_w, _), dtype=int)
    # image[:, :xmin, :] = np.zeros((img_h, xmin, _), dtype=int)
    # image[:, xmax+1:img_w, :] = np.zeros((img_h, img_w-xmax-1, _), dtype=int)
    new_image = mask_image[ymin:ymax, xmin:xmax, :]
    return new_image


# mask only bounding box
def crop_image1(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                score_threshold=0.6, iou_threshold=0.7, resize=False, target_size=(224, 224, 3), do_return=False):
    original_image = cv2.imread(image_path)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='soft-nms')
    new_image = prepare_image1(original_image, bboxes)
    if resize:
        new_image = image_preprocess1(new_image, target_size)
    if output_path != '':
        try:
            cv2.imwrite(output_path, new_image)
        except:
            pass
    if do_return:
        return new_image


def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES,
                 score_threshold=0.5, iou_threshold=0.3, rectangle_colors=''):
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='soft-nms')

    image, string_char = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
    # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

    if output_path != '':
        cv2.imwrite(output_path, image)
    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()
        
    return image, string_char


def detect_bboxes(Yolo, image_path, input_size=416, score_threshold=0.6, iou_threshold=0.3):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)

    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')
    return bboxes


def postprocess_mp(Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES, score_threshold, iou_threshold, rectangle_colors, realtime):
    times = []
    while True:
        if Predicted_data.qsize()>0:
            pred_bbox = Predicted_data.get()
            if realtime:
                while original_frames.qsize() > 1:
                    original_image = original_frames.get()
            else:
                original_image = original_frames.get()
            
            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')
            image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            times.append(time.time()-Processing_times.get())
            times = times[-20:]
            
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            #print("Time: {:.2f}ms, Final FPS: {:.1f}".format(ms, fps))
            
            Processed_frames.put(image)

