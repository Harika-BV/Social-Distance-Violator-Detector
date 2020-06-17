# Social-Distance-Violator-Detector
Detect the social distance violations in real time using OpenCV and deep learning

> ## ***"Social distancing is the only way to counter Covid-19"***

**Requirements**
- Python
- Numpy
- OpenCV
- Scipy
- Object Detection models (Mobilenet or Googlenet or YOLO)

***Steps***
1. Input the video stream using OpenCV
2. Detect the people in the video stream using any of the object detection models
3. Find the centroids of the identified people and calculate the distance between the centroids
4. Check if the distance calculated above is lesser than the minimum value (in pixels) and consider it as violation
5. Draw the bounding boxes in red color around the person if violating, green color box if the person is following the norms
