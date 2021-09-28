import numpy as np
import time
import cv2
import os

def resize(image, width=None, height=None):
    if (width is None) & (height is None):
        raise Exception("Height and Width npth are None")
    elif (width is not None) & (height is not None):
        raise Exception("You haved passed npth Height and Width both value")
    elif (width is not None) & (height is None):
        h, w, c = image.shape
        height = int((h / w) * width)
        return cv2.resize(image, (width, height))
    elif (width is None) & (height is not None):
        h, w, c = image.shape
        width = int((w / h) * height)
        return cv2.resize(image, (width, height))


def detect(image, net, ln, Labels, colors, drawBox=True, return_cords=False, minConfi=0.01, thresh=0.1):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence >= minConfi:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, minConfi, thresh)

    coords_boxes = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            color = [int(c) for c in colors[classIDs[i]]]

            if drawBox:
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.4f}".format(Labels[classIDs[i]], confidences[i])
                # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if return_cords:
                coords_boxes.append([Labels[classIDs[i]], confidences[i], x, y, x+w, y+h])

    if drawBox and return_cords:
        return image, coords_boxes
    elif drawBox and not return_cords:
        return image
    elif not drawBox and return_cords:
        return coords_boxes


def sort_words(boxes):
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)
    
    boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []
    for box in boxes:
        if box[1] > current_line + mean_height:
            lines.append(tmp_line)
            tmp_line = [box]
            current_line = box[1]            
            continue
        tmp_line.append(box)
    lines.append(tmp_line)
        
    for line in lines:
        line.sort(key=lambda box: box[0])
        
    return lines