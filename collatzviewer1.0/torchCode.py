import torch
import numpy as np
import pngExport1 as png1
import time

from PIL import Image
from PIL import ImageDraw

torch.set_default_dtype(torch.float64)

def collatzAlgorithmSimple(a):
	k=1
	while a!=1:
		k+=1
		if a % 2 == 0: #a is even
			a=a/2
		else: #a is odd
			a=a*3+1
	return k


def hsv_to_rgb255(h, s, v):
            if s == 0.0: v*=255; return (v, v, v)
            i = int(h*6.) # XXX assume int() truncates!
            f = (h*6.)-i; p,q,t = int(255*(v*(1.-s))), int(255*(v*(1.-s*f))), int(255*(v*(1.-s*(1.-f)))); v*=255; i%=6
            if i == 0: return (v, t, p)
            if i == 1: return (q, v, p)
            if i == 2: return (p, v, t)
            if i == 3: return (p, q, v)
            if i == 4: return (t, p, v)
            if i == 5: return (v, p, q)


def generateImage(xn, xc1MaxPoints, xc2MaxPoints, xcenterX, xcenterY, xoffset, xtextOnImages):

	if torch.cuda.is_available():
		device = torch.device("cuda") # for multiple GPUs, add "cuda:0" or "cuda:1" etc
	else:
		device = torch.device("cpu")


	stepsEpsilon=0.00000000000001#for torch.arange non-integer step is subject to floating point rounding errors when comparing against end; to avoid inconsistency, add a small epsilon to end in such cases.
	#https://stackoverflow.com/questions/9528421/value-for-epsilon-in-python

	useMinusHue=False

	n=xn
	centerX=xcenterX
	centerY=xcenterY
	offset=xoffset
	c1MaxPoints = xc1MaxPoints
	c2MaxPoints = xc2MaxPoints
	textOnImages = xtextOnImages

	c1min = centerX-offset
	c1max = centerX+offset
	c2min = centerY-offset
	c2max = centerY+offset
	c1step = (c1max - c1min)/(c1MaxPoints - 1)
	c2step = (c2max - c2min)/(c2MaxPoints - 1)


	A006577Value=collatzAlgorithmSimple(n)
	maxIterationsFactor = 5
	maxIterations=A006577Value*maxIterationsFactor

	c1=torch.arange(c1min,c1max+stepsEpsilon,c1step)
	c2=torch.arange(c2min,c2max+stepsEpsilon,c2step)
	c1x=c1.repeat(len(c2), 1)
	c2y=c2.repeat(len(c1), 1)
	c2y=torch.transpose(c2y,0,1)

	width=c1MaxPoints
	height=c2MaxPoints
	width=len(c1)
	height=len(c2)

	collatzArray= [[n]*height]*width
	tensor1=torch.tensor(collatzArray) #convert list or array to tensor     
	tensor2 = torch.tensor(collatzArray, dtype = torch.float64)

	kArray= [[1]*height]*width
	ktensor=torch.tensor(kArray) #convert list or array to tensor

	if device==torch.device("cuda"):

		tensor2_gpu = tensor2.to(device)
		tensor2old_gpu = tensor2.to(device)
		c1x_gpu = c1x.to(device)
		c2y_gpu = c2y.to(device)
		ktensor_gpu = ktensor.to(device)
		maxIterationsTemp=torch.as_tensor(maxIterations)
		maxIterations_gpu=maxIterationsTemp.to(device)
		torch.cuda.synchronize() #Waits for all kernels in all streams on a CUDA device to complete

		startOuterLoop=time.time()

		for i in range(maxIterations_gpu):

			start = time.time()

			tensor2old_gpu=tensor2_gpu
			tensor2_gpu = torch.where((tensor2_gpu>1) & (ktensor_gpu<=maxIterations_gpu), (torch.where(tensor2_gpu % 2 == 0, torch.round((tensor2_gpu / 2)*c1x_gpu), torch.round((tensor2_gpu * 3 + 1)*c2y_gpu))), tensor2_gpu) #don't assign tensors to 0.0 or 1.0 if first conditionals (tensor2>1) & (ktensor<=maxIterations) not met
			torch.cuda.synchronize()

			ktensor_gpu = torch.where((tensor2old_gpu>1) & (ktensor_gpu<=maxIterations_gpu), ktensor_gpu+1, ktensor_gpu)
			torch.cuda.synchronize()

		outerLoopTime=time.time() - startOuterLoop

		image2 = np.zeros((height, width, 4), dtype=np.uint8)

		ktensor = ktensor_gpu.cpu() #use tensor.cpu() to copy the tensor to host memory before converting to numpy array	
		ktensorCopyTemp=ktensor
		collatzArrayDataCopyTemp=ktensorCopyTemp.numpy()
		collatzArrayData=collatzArrayDataCopyTemp.copy()
		collatzArrayData3=np.interp(collatzArrayData, (collatzArrayData.min(),collatzArrayData.max()), (0, 0.9))

		startOuterLoopImage=time.time()
		for x in range(width):
					for y in range(height):
						hue = collatzArrayData3[x,y] #use for c1,c2 max at top right of image
						temp=collatzArrayData[x,y]
						
						if temp==A006577Value:
							r,g,b = hsv_to_rgb255(0,0,1) # x,0,1 is white
						else:
							if temp<maxIterations:
								hue=temp/maxIterations
								if useMinusHue:
									r,g,b = hsv_to_rgb255(-hue,1,1) # -hue gives different colours
								else:
										r,g,b = hsv_to_rgb255(hue,1,1) # x,1,1 gives a scaled colour
							else:
								r,g,b = hsv_to_rgb255(0,1,0) # x,1,0 is black
								
						image2[y, x, 0]= r
						image2[y, x, 1] = g
						image2[y, x , 2]= b
						
		image2[:, :, 3] = 255 #set alpha full opacity
		string1=f"collatz pytorch cuda test n{n} width{width} height{height} centerX{centerX} centerY{centerY} offset{offset} time{round(outerLoopTime,3)}s.png"
		string1="image.png"
		png1.write_png(image2, string1)

		if textOnImages==True:
			img = Image.open('image.png')
			I1 = ImageDraw.Draw(img)
			I1.text((28, 36), f"n {n}", fill=(255, 255, 255))
			I1.text((28, 46), f"centerX {centerX}", fill=(255, 255, 255))
			I1.text((28, 56), f"centerY {centerY}", fill=(255, 255, 255))
			I1.text((28, 66), f"offset {offset}", fill=(255, 255, 255))
			I1.text((28, 76), f"width {width}", fill=(255, 255, 255))
			I1.text((28, 86), f"height {height}", fill=(255, 255, 255))
			I1.text((28, 96), f"cuda time {round(outerLoopTime,3)}s", fill=(255, 255, 255))
			img.save("image.png")

		imageWriteTime=time.time() - startOuterLoopImage

	if device==torch.device("cpu"):

		startOuterLoop=time.time()

		for i in range(maxIterations):

			start = time.time()

			tensor2old=tensor2
			
			tensor2 = torch.where((tensor2>1) & (ktensor<=maxIterations), (torch.where(tensor2 % 2 == 0, torch.round((tensor2 / 2)*c1x), torch.round((tensor2 * 3 + 1)*c2y))), tensor2) #don't assign tensors to 0.0 or 1.0 if first conditionals (tensor2>1) & (ktensor<=maxIterations) not met
			
			ktensor = torch.where((tensor2old>1) & (ktensor<=maxIterations), ktensor+1, ktensor)
			
		outerLoopTime=time.time() - startOuterLoop
		
		image2 = np.zeros((height, width, 4), dtype=np.uint8)

		ktensorCopyTemp=ktensor
		collatzArrayDataCopyTemp=ktensorCopyTemp.numpy()
		collatzArrayData=collatzArrayDataCopyTemp.copy()
		collatzArrayData3=np.interp(collatzArrayData, (collatzArrayData.min(),collatzArrayData.max()), (0, 0.9))

		startOuterLoopImage=time.time()
		for x in range(width):
					for y in range(height):
						hue = collatzArrayData3[x,y] #use for c1,c2 max at top right of image
						temp=collatzArrayData[x,y]
						
						if temp==A006577Value:
							r,g,b = hsv_to_rgb255(0,0,1) # x,0,1 is white
						else:
							if temp<maxIterations:
								hue=temp/maxIterations
								if useMinusHue:
									r,g,b = hsv_to_rgb255(-hue,1,1) # -hue gives different colours
								else:
										r,g,b = hsv_to_rgb255(hue,1,1) # x,1,1 gives a scaled colour
							else:
								r,g,b = hsv_to_rgb255(0,1,0) # x,1,0 is black
								
						image2[y, x, 0]= r
						image2[y, x, 1] = g
						image2[y, x , 2]= b
						
		image2[:, :, 3] = 255 #set alpha full opacity
		string1=f"collatz pytorch cpu test n{n} width{width} height{height} centerX{centerX} centerY{centerY} offset{offset} time{round(outerLoopTime,3)}s.png"
		string1="image.png"
		png1.write_png(image2, string1)

		if textOnImages==True:
			img = Image.open('image.png')
			I1 = ImageDraw.Draw(img)
			I1.text((28, 36), f"n {n}", fill=(255, 255, 255))
			I1.text((28, 46), f"centerX {centerX}", fill=(255, 255, 255))
			I1.text((28, 56), f"centerY {centerY}", fill=(255, 255, 255))
			I1.text((28, 66), f"offset {offset}", fill=(255, 255, 255))
			I1.text((28, 76), f"width {width}", fill=(255, 255, 255))
			I1.text((28, 86), f"height {height}", fill=(255, 255, 255))
			I1.text((28, 96), f"cuda time {round(outerLoopTime,3)}s", fill=(255, 255, 255))
			img.save("image.png")

		imageWriteTime=time.time() - startOuterLoopImage


