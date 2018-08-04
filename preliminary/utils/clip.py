#coding: utf-8

from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np

ImgPath = 'C:/Users/18292/Desktop/xuelang_dataset/process/jpg_2/' 
AnnoPath = 'C:/Users/18292/Desktop/xuelang_dataset/process/xml_2/'
ProcessedPath = 'C:/Users/18292/Desktop/xuelang_dataset/process/Cropedflaw_2/'

imagelist = os.listdir(ImgPath)
for image in imagelist:
	image_pre, ext = os.path.splitext(image)
	imgfile = ImgPath + image 
	xmlfile = AnnoPath + image_pre + '.xml'
	DomTree = xml.dom.minidom.parse(xmlfile)
	annotation = DomTree.documentElement
	filenamelist = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]
	filename = filenamelist[0].childNodes[0].data
	objectlist = annotation.getElementsByTagName('object')
	i = 1
	for objects in objectlist:
		namelist = objects.getElementsByTagName('name')
		objectname = namelist[0].childNodes[0].data
		savepath = ProcessedPath + objectname
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		bndbox = objects.getElementsByTagName('bndbox')
		cropboxes = []
		for box in bndbox:
			x1_list = box.getElementsByTagName('xmin')
			x1 = int(x1_list[0].childNodes[0].data)
			y1_list = box.getElementsByTagName('ymin')
			y1 = int(y1_list[0].childNodes[0].data)
			x2_list = box.getElementsByTagName('xmax')
			x2 = int(x2_list[0].childNodes[0].data)
			y2_list = box.getElementsByTagName('ymax')
			y2 = int(y2_list[0].childNodes[0].data)
			w = x2 - x1
			h = y2 - y1
			obj = np.array([x1,y1,x2,y2])
			shift = np.array([[0.8,0.8,1.2,1.2],[0.9,0.9,1.1,1.1],[1,1,1,1],[0.7,0.7,1,1],[1,1,1.2,1.2],\
				[0.7,1,1,1.2],[1,0.7,1.2,1],[(x1+w*1/3)/x1,(y1+h*1/3)/y1,(x2+w*1/3)/x2,(y2+h*1/3)/y2],\
				[(x1-w*1/3)/x1,(y1-h*1/3)/y1,(x2-w*1/3)/x2,(y2-h*1/3)/y2]])
			XYmatrix = np.tile(obj,(9,1))  
			cropboxes = XYmatrix * shift
			img = Image.open(imgfile)
			for cropbox in cropboxes:
				cropedimg = img.crop(cropbox)
				cropedimg.save(savepath + '/' + image_pre + '_' + str(i) + '.jpg')
				i += 1
