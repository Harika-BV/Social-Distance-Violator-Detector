# import the necessary packages

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from scipy.spatial import distance as dist

display = 1						# 0 if you don't want the video to be shown while processing, else put > 0
output = "output_final.avi"		# Output video path

# The below function helps in detecting the objects (Person) in the video frame

def detect_people(frame):
	results = []
	boxes = []
	centroids = []
	confidences = []
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	
	net.setInput(blob)
	detections = net.forward()
	
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]
		
		if confidence > args["confidence"]:
			try:
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				y = startY - 15 if startY - 15 > 15 else startY + 15
				
				x1 = int((startX  + endX) / 2)
				y1 = int((startY  + endY) / 2)
					
				boxes.append([int(startX), int(startY),int(endX), int(endY)])
				centroids.append((x1, y1))
				confidences.append(float(confidence))
				r = (confidences[i], (startX, startY, endX, endY), centroids[i])
				results.append(r)
			
			except:
				continue
				
	return results
	
	
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", required=True, help="Video Path")
args = vars(ap.parse_args())

CLASSES = ["person"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None
time.sleep(2.0)
fps = FPS().start()

while True:

	ret, frame = vs.read()
	frame = imutils.resize(frame, width=500)
	results = detect_people(frame)
	violate = set()	
	
	if len(results) >= 2:
		centroids = np.array([r[2] for r in results])
		#Euclidean Distance
		D = dist.cdist(centroids, centroids, metric="euclidean")

		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < 50:
					violate.add(i)
					violate.add(j)
	
	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 0, 255)
		
		if i in violate:
			color = (0, 0, 255)
		
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
	text = "Violation Detections: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
	
	if display > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	if output != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output, fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)
		
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
