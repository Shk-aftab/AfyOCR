import os
import cv2
import numpy as np
from utils.detect import *

def decodeText(scores):
	text = ""
	alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
	for i in range(scores.shape[0]):
		c = np.argmax(scores[i][0])
		if c != 0:
			text += alphabet[c - 1]
		else:
			text += '-'

	# adjacent same letters as well as background text must be removed to get the final output
	char_list = []
	for i in range(len(text)):
		if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
			char_list.append(text[i])
	return ''.join(char_list)

def inference(image_path):
	try:	
		image = cv2.imread(image_path)
		image = resize(image, height=1280)
		image2 = image.copy()
		recognizer = cv2.dnn.readNetFromONNX("model/CRNN_VGG_BiLSTM_CTC.onnx")
		net = cv2.dnn.readNetFromDarknet("model/yolov4-custom.cfg","model/yolov4-custom_best.weights")

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		drawed, coords = detect(image2.copy(), net, ln, ["text"], [(255, 0, 0)], return_cords=True)
		boxes = np.array([coord[2:] for coord in coords]).astype(np.int64)
		lines = sort_words(boxes)

		answer = ""
		for line in lines:
			for coord in line:
				coord = np.maximum(coord, 0)
				cropped = image2[coord[1]: coord[3], coord[0]: coord[2]]

				cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
				blob = cv2.dnn.blobFromImage(cropped, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)
				recognizer.setInput(blob)
				result = recognizer.forward()
				wordRecognized = decodeText(result)

				answer += " " + wordRecognized

				drawed = cv2.putText(drawed, wordRecognized, (coord[0], coord[1] - 3), 
					cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
					
		cv2.imwrite(image_path, drawed)
		print(answer)
		return answer
	except:
		cv2.imwrite(image_path, drawed)
		answer = 'Unable to Detect or Recognize the text from image'
		return answer