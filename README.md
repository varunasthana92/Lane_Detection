# Lane Detection

Objective of the project is to detect the drive lane on the given data set (a video). Project was part of the academic coursework at the Univeristy of Maryland-College Park.

## Pipe Line
<p align="left">
1) Convert the scene into bird-eye view with a Homography.<br/>
2) Trim the field of view according to camera position to get area of interest (i.e. road).<br/>
<p align="center">
<img src="https://github.com/varunasthana92/Lane_Detection/blob/master/pics/bird_view.jpg" >
</p>
3) Create a binary image from the trimmed image. White and yellow pixels are retianed (for white and yellow lanes).<br/>
<p align="center">
<img src="https://github.com/varunasthana92/Lane_Detection/blob/master/pics/Binary.jpg">
</p>
4) With histogram along the width of the image, detect the left and right lane start points at the bottom of the image (mean of pixel x values). Left lane point will be in left half width and right lane point will be in the right half width (assuming camera center to be at the mid of the image).<br/>
<p align="center">
<img src="https://github.com/varunasthana92/Lane_Detection/blob/master/pics/findStart.jpg">
</p>
5) Divide the height of the binary image into defined number of boxes. Detetct x-mean of white pixels in each successive box to track the lane. Once all center points of left and right lanes are detected, fit a line from these pixel points to get the missing points between lane patches (for continuous lane detection).<br/>
<p align="center">
<img src="https://github.com/varunasthana92/Lane_Detection/blob/master/pics/lanes.jpg">
</p>
6) With inverse homography, plot these detected points.<br/>
<p align="center">
<img src="https://github.com/varunasthana92/Lane_Detection/blob/master/pics/FinalImage.jpg">
</p>
</p>

### Dependencies
- python3
- opencv
- numpy
- matplotlib


Use the below command to run the lane-detector on the provided input video.<br/>
--saveImg : Set 1 to save pipeline processing images. Default 0<br/>
--saveVideo : Set 1 to save final output as a video. Default 0
```
python3 detect.py --saveImgs=1 --saveVideo=1
```
