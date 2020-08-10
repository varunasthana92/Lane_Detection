# Lane Detection

Project was part of the academic coursework at the Univeristy of Maryland-College Park. The prpose of the project is to detect the drive lane on the given data set (a video).

## Pipe Line
1) Convert the scene into bird-eye view with a Homography.\
<p align="center">
<img src="https://github.com/varunasthana92/Human_Detection_OpenCV/blob/master/additional_files/expected_behaviour.png">
</p>
2) Trim the field of view according to camera position to get area of interest (i.e. road).\
3) Create a binary image from the trimmed image. White and yellow pixels are retianed (for white and yellow lanes).\
4) With histogram along the width of the image, detect the left and right lane pooints. Left lane point will be in left half width and right lane points will be in right half widht, assuming camera center to be at the mid of the image.\
5) Divide the height of the binary image into defined number of boxes. Detetct white pixels in each successive patch to track the lane. Once all center points of left and right lanes are detected, fit a line from these points to get the missing points between lane patches (for continuous lane detection).\
6) With inverse homography, plot these detected points

### Dependencies
- python3
- opencv
- numpy
- matplotlib


Use the below command to run the lane-detector on the provided input video.\
--saveImg : Set 1 to save pipeline processing images. Default 0\
--saveVideo : Set 1 to save final output as a video. Default 0
```
python3 detect.py --saveImgs=1 --saveVideo=1
```
