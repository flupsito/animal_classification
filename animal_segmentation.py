import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def mask_animal(img):
	kernel = np.ones((5,5),np.float32)
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
	rows, cols, _ = lab.shape
	bg_roi = lab[-50:, -50:, :]
	bg_pixels = bg_roi.reshape(-1, 3).astype(np.float32)
	mean_vec = np.mean(bg_pixels, axis=0)
	cov_matrix = np.cov(bg_pixels, rowvar=False)
	inv_cov_matrix = np.linalg.inv(cov_matrix)
	flat_pixels = lab.reshape(-1, 3).astype(np.float32)
	diff = flat_pixels - mean_vec
	mdist = np.sqrt(np.sum((diff @ inv_cov_matrix) * diff, axis=1))
	mdist_img = mdist.reshape(rows, cols)
	threshold_val = 10.0
	mask = (mdist_img > threshold_val).astype(np.uint8) * 255
	eroded = cv2.erode(mask, kernel)
	closed = cv2.dilate(eroded, kernel)
	cv2.imwrite("mask_animal.jpg", closed)
	return closed

def shape_detection(org):
	img = cv2.imread("mask_animal.jpg")
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(grey, 0.9, 1, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contour_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
	return (contour_sorted)

def shape_properties(contour):
	return cv2.contourArea(contour), cv2.arcLength(contour, True)

def image_properties(org):
	contours = shape_detection(org)
	parameter = {}
	for i in range(min(1, len(contours))):
		area, perimeter = shape_properties(contours[i])
		parameter[i] = (cv2.moments(contours[i]), {"Area": area}, {"Perimeter": perimeter})
	return parameter

def get_label(key):
	if "antilope" in key:
		return 0
	elif "lion" in key:
		return 1
	elif "elefant" in key:
		return 2
	elif "zebra" in key:
		return 3
	elif "rhino" in key:
		return 4
	else:
		raise ValueError (f"wrong key {key}")
	
def get_name(label):
	if label == 0:
		return ("antilope")
	elif label == 1:
		return ("lion")
	elif label == 2:
		return ("elefant")
	elif label == 3:
		return ("zebra")
	elif label == 4:
		return ("rhino")
	else:
		raise ValueError (f"wrong label {label}")

def plot(params, label):
	fig = plt.figure()
	s = plt.scatter(params[:,1], params[:,0], c=label)
	plt.xlabel("Perimeter")
	plt.colorbar(s)
	plt.ylabel("Area")
	plt.show()

def show_params(parameter):
	params = np.zeros(shape=(98 * 1, 2))
	label = []
	keys = parameter.keys()
	x = 0
	for key in keys:
		l = get_label(key)
		img_para = parameter[key]
		ks = img_para.keys()
		for i, _ in enumerate(ks):
			para = img_para[i]
			params[x][0] = para[1]["Area"]
			params[x][1] = para[2]["Perimeter"]
			label.append(l)
			x += 1
	plot(params, label)
	return params, label

def get_text_coordinates(cnt):
	M = cv2.moments(cnt)
	if M["m00"] != 0:
		cx = int(M["m10"] / M["m00"])
		cy = int(M["m01"] / M["m00"])
	else:
		x, y, w, h = cv2.boundingRect(cnt)
		cx, cy = x + w // 2, y + h // 2
	return cx, cy

def animal_prediction(data, label):
	svm = SVC(kernel='rbf')
	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data)
	svm.fit(data_scaled, label)
	img_pred = os.listdir("capture_19")
	pred_para = np.ndarray(shape=(1,2))
	for i in img_pred:
		pred = np.zeros(shape=(5,1))
		img_to_pred = cv2.imread(os.path.join("capture_19", i))
		if img_to_pred is None:
			continue
		mask_animal(img_to_pred)
		contour_sorted = shape_detection(img_to_pred)
		for j in range(min(5, len(contour_sorted))):
			pred_para[0][0], pred_para[0][1] = shape_properties(contour_sorted[j])
			pred_para_scaled = scaler.transform(pred_para)
			pred[j] = svm.predict(pred_para_scaled)
		random.seed(12345)
		for idx in range(min(5, len(contour_sorted))):
			if np.round(cv2.arcLength(contour_sorted[idx], True), 2) < 800:
				continue
			cv2.drawContours(img_to_pred, contour_sorted, idx, (255,0,0), 2, 8)
			cx, cy = get_text_coordinates(contour_sorted[idx])
			cv2.putText(img_to_pred, get_name(pred[idx][0]), (cx + 100, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 3, cv2.LINE_AA)
		cv2.imshow("Contours", img_to_pred)
		cv2.waitKey(-1)

if __name__ == "__main__":
	imgs = os.listdir("animal")
	params = {}
	for i, file_name in enumerate(imgs):
		if file_name != "gps_locations.log" and file_name != ".DS_Store":
			img = cv2.imread(os.path.join("animal", file_name))
			if img is None:
				raise ValueError ("img is none")
			mask = mask_animal(img)
			params[file_name] = image_properties(img)
	data, label = show_params(params)
	animal_prediction(data=data, label=label)