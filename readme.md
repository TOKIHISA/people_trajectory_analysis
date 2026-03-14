# Pedestrian Trajectory Mapping with MapLibre and OpenCV from Smartphone Videos
## Summary
I needed to analyze pedestrian flow patterns and draw them on map in urban spaces, so I built a simple pipeline to extract trajectories from smartphone videos which shot from a low line of sight.
## Overall Pipeline
1. Capturing videos of pedestrians with smartphone
2. Pedestrian Detection with openCV DNN and makeing trajectories with CentroidTracker
3. Homography transformation from video coordinate to wgs84
4. Draw trajectory on interactive map

## In this session
I will talk about the barriers encountered during the experiment.
1. From holding the smartphone with my hand to securing it with a tripod.
2. How effective is a low line of sight in videos.
3. Comparison between centroid and footprint trackers.
4. How and how many trajectory points I sampled.


## Technologies Used
This project utilizes only open source technologies:
- [**OpenCV DNN**](https://opencv.org/) - Computer vision and deep learning
- [**CentroidTracker**](https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) - Multi-object tracking
- [**MapLibre GL**](https://maplibre.org/) - Interactive web mapping
- [**YOLOx**](https://github.com/Megvii-BaseDetection/YOLOX) - Object detection
- [**Python**](https://www.python.org/), [**TypeScript**](https://www.typescriptlang.org/), [**React**](https://react.dev/) - Development frameworks

## License
This project acknowledges the following open source licenses:
- [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) - OpenCV, YOLOx, TypeScript
- [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) - MapLibre GL
- [MIT](https://opensource.org/licenses/MIT) - React
- [Python Software Foundation License](https://docs.python.org/3/license.html) - Python

## 
I expect that this project contributes to individual protest analysis by analyzing pedestrian reactions.