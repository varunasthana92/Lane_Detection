import sys
# sys.path.remove(sys.path[1])
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import argparse


def histEqu(img):
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')
	img2 = cdf[img]
	h,w = img2.shape
	return img2.reshape([h,w,1])

def binary(img):
	newImg = np.dstack([histEqu(img[:,:,0]), histEqu(img[:,:,1]), histEqu(img[:,:,2])])
	blur = cv2.GaussianBlur(newImg,(5,5),0)
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([20, 100, 100], dtype = np.uint8)
	upper_yellow = np.array([30, 255, 255], dtype = np.uint8)
	mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
	mask_white = cv2.inRange(gray, 250, 255)
	mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
	mask_yw_image = cv2.bitwise_and(gray, mask_yw)
	return mask_yw_image

def undistort(img, mtx, dist):
	undistort = cv2.undistort(img, mtx, dist, None, mtx)
	return undistort

def prepresocess(img):
	blur = cv2.GaussianBlur(img,(3,3),0)
	edges = cv2.Canny(blur,300,300)
	return edges

def warpImage(img, H):
	h,w,_ = img.shape
	return cv2.warpPerspective(img, H, (w,h))

def getLaneStart(warpedBin, mid, imgShow):
	h,w = warpedBin.shape
	hist = warpedBin.sum(axis=0)
	leftStart = np.argmax(hist[:mid])
	rightStart = np.argmax(hist[mid:])
	leftStartPoint = (leftStart, h)
	leftEndPoint = (leftStart, 0)
	rightStartPoint = (mid+rightStart, h)
	rightEndPoint = (mid+rightStart, 0)
	# imgShow = warpedBin.copy()
	color = (0,0,255)
	thickness = 7
	imgShow = cv2.line(imgShow, leftStartPoint, leftEndPoint, color, thickness)
	imgShow = cv2.line(imgShow, rightStartPoint, rightEndPoint, color, thickness)
	return imgShow, leftStart, rightStart+mid

def rect(img, maxBoxes, leftCentre, rightCentre, warpedBin):
	h,w,_ = img.shape
	image = img.copy()
	h_box = h//maxBoxes
	color = (0,255,0)
	thickness = 3
	leftLanePoints = []
	rightLanePoints = []
	margin = 100
	for i in range(maxBoxes):
		left_start_point = (leftCentre - margin , h_box*(maxBoxes - i - 1))
		left_end_point = (leftCentre + margin, h_box*(maxBoxes - i))
		right_start_point = (rightCentre - margin, h_box*(maxBoxes - i - 1))
		right_end_point = (rightCentre + margin, h_box*(maxBoxes - i))

		small_patch = warpedBin[left_start_point[1]:left_end_point[1], left_start_point[0]:left_end_point[0]]
		# maximum_in_patch = small_patch.max(axis=0)
		y,x = small_patch.nonzero() 
		if len(x)>15:
			x = x + left_start_point[0] 
			y = y + left_start_point[1]
			for xx in range(len(x)):
				image = cv2.circle(image, (x[xx],y[xx]), 1, color, -1) 
				leftLanePoints.append((x[xx],y[xx]))
			#### set new centre
			leftCentre = int(x.mean())
			
		small_patch = warpedBin[right_start_point[1]:right_end_point[1], right_start_point[0]:right_end_point[0]]
		y,x = small_patch.nonzero()
		if len(x)>15:
			x = x + right_start_point[0] 
			y = y + right_start_point[1]
			for xx in range(len(x)):
				image = cv2.circle(image, (x[xx],y[xx]), 1, color, -1) 
				rightLanePoints.append((x[xx], y[xx]))
			### Set new right centre
			rightCentre = int(x.mean())

		image = cv2.rectangle(image, left_start_point, left_end_point, color, thickness) 
		image = cv2.rectangle(image, right_start_point, right_end_point, color, thickness)
		


	l_x, l_y = [i[0] for i in leftLanePoints], [i[1] for i in leftLanePoints]
	r_x, r_y = [i[0] for i in rightLanePoints], [i[1] for i in rightLanePoints]
	leftLane = []
	rightLane = []
	try:
		L = np.polyfit(l_y, l_x, 2)
		LP = np.poly1d(L)
		R = np.polyfit(r_y, r_x, 2)
		LR = np.poly1d(R)
		draw_x = list(range(0, h))
		draw_yl = [LP(i) for i in draw_x]
		draw_yr = [LR(i) for i in draw_x]
		for i_x in draw_x:
			leftLane.append((int(LP(i_x)), i_x))
			rightLane.append((int(LR(i_x)), i_x))
			image = cv2.circle(image, (int(LP(i_x)), i_x), 3, color, -1)
			image = cv2.circle(image, (int(LR(i_x)), i_x), 3, color, -1)
	except:
		pass
	return image, leftLane, rightLane


def main(Args):
	VideoPath = Args.VideoPath
	saveImg = Args.saveImgs
	saveVideo = Args.saveVideo

	K = np.array([[9.037596e+02, 0.000000e+00, 6.957519e+02],
				  [0.000000e+00, 9.019653e+02, 2.242509e+02],
				  [0.000000e+00, 0.000000e+00, 1.000000e+00]])

	D = np.array([[-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02]])

	Hpoints = np.array([[275,473],[886,473],[819,394],[401,394]])
	box_points = np.array([[275,473],[886,473],[886,394],[275,394]]) # bird-eye view world points of a sqaure
	H = cv2.findHomography(Hpoints, box_points) # finding the homography matrix from image to world (bird-eye)
	Hinv = np.linalg.inv(H[0]) # inverse of homography
	plt.ion()
	mid = 600
	if(saveVideo):
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		vw = cv2.VideoWriter("data_output.avi", fourcc, 30, (1392, 512))
	

	cap = cv2.VideoCapture(VideoPath)
	while (cap.isOpened()):	
		ret, img1 = cap.read()
		if(not ret):
			break
		img1 = undistort(img1, K, D)
		img = warpImage(img1, H[0])
		if saveImg:
			cv2.imwrite("pics/Unwarped.jpg",img)
		img_ = binary(img)
		if saveImg:
			cv2.imwrite("pics/Binary.jpg",img_)
		# cv2.imshow("Warped", img_)
		_, leftStart, rightStart = getLaneStart(img_, 600, img.copy())
		if saveImg:
			cv2.imwrite("pics/findStart.jpg",_)
		# cv2.imshow("Lane", getLaneStart(img_, 600, img.copy()))
		maskedImage, leftLane_, rightLane_ = rect(img, 16, leftStart, rightStart, img_)
		
		if saveImg:
			cv2.imwrite("pics/boxes.jpg", maskedImage)
			cv2.imwrite("pics/lanes.jpg",maskedImage)
		#############################################################
		# Following line overlays transparent rectangle over the image

		image = img1.copy()
		if len(leftLane_)!=0 and len(rightLane_)!=0:
			leftLane__ = np.array(leftLane_).T
			leftLane__ = np.vstack((leftLane__, np.ones([1, leftLane__.shape[1]])))
			leftLane = Hinv.dot(leftLane__)
			leftLane[0,:] = leftLane[0,:]//leftLane[2,:]
			leftLane[1,:] = leftLane[1,:]//leftLane[2,:]
			leftLane[2,:] = leftLane[2,:]//leftLane[2,:]

			rightLane__ = np.array(rightLane_).T
			rightLane__ = np.vstack((rightLane__, np.ones([1, rightLane__.shape[1]])))
			rightLane = Hinv.dot(rightLane__)
			rightLane[0,:] = rightLane[0,:]//rightLane[2,:]
			rightLane[1,:] = rightLane[1,:]//rightLane[2,:]
			rightLane[2,:] = rightLane[2,:]//rightLane[2,:]

			y_left_min_index, y_left_max_index = np.argmin(leftLane[1,:]), np.argmax(leftLane[1,:])
			y_right_min_index, y_right_max_index = np.argmin(rightLane[1,:]), np.argmax(rightLane[1,:]) 

			turn_left = leftLane[0, y_left_min_index] - leftLane[0, y_left_max_index]
			turn_right = rightLane[0, y_right_min_index] - rightLane[0, y_right_max_index]
			turn = turn_left/turn_right	
			if turn>0 and np.abs(turn_left) > 500:
				font = cv2.FONT_HERSHEY_SIMPLEX 
				org = (50, 50) 
				fontScale = 1
				color_ = (0, 0, 255)  
				thickness_ = 4   
				img1 = cv2.putText(image, 'Left turn', org, font,  
	                   fontScale, color_, thickness_, cv2.LINE_AA)
			elif turn<0 and np.abs(turn_right) > 500:
				font = cv2.FONT_HERSHEY_SIMPLEX 
				org = (50, 50) 
				fontScale = 1
				color_ = (0, 0, 255) 
				thickness_ = 4
				img1 = cv2.putText(image, 'Right turn', org, font,  
	                   fontScale, color_, thickness_, cv2.LINE_AA)
			else:
				font = cv2.FONT_HERSHEY_SIMPLEX 
				org = (50, 50) 
				fontScale = 1
				color_ = (0, 0, 255) 
				thickness_ = 4  
				img1 = cv2.putText(image, 'Straight', org, font,  
	                   fontScale, color_, thickness_, cv2.LINE_AA)

			for i in range(min(leftLane.shape[1], rightLane.shape[1])):
				try:
					image[int(leftLane[1,i]), int(leftLane[0,i]):int(rightLane[0,i]),:] = np.array([0,255,0])
				except:
					pass
		alpha = 0.4  # Transparency factor.
		image_new = cv2.addWeighted(image, alpha, img1, 1 - alpha, 0)
		cv2.imshow("T",image_new)
		if saveImg:
			cv2.imwrite("pics/FinalImage.jpg", image_new)
		if(saveVideo):
			vw.write(image_new)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	plt.ioff()
	if(saveVideo):
		vw.release()
	cv2.destroyAllWindows()
	# cap.release()


if __name__ == '__main__':
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--VideoPath', default="./data/data.avi", help='Path of the input video')
	Parser.add_argument('--saveVideo', type=int, default= 0, help='Set 1 to save final output as a video')
	Parser.add_argument('--saveImgs', type=int, default= 0, help='Set 1 to save pipeline processing images')
	Args = Parser.parse_args()
	main(Args)
