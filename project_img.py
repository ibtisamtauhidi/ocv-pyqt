from PyQt4 import QtCore, QtGui, uic
import sys
import cv2
import numpy as np
import threading
import time
import Queue
import easygui
import os
from matplotlib import pyplot as plt


running = False
capture_thread = None
form_class = uic.loadUiType("simple.ui")[0]
q = Queue.Queue()
canny = False
laplacian = False
sobel = False
sobel_x = False
sobel_y = False
source = False
contrast = False
image_f = False
dft = False
def grab(file_name, queue, width, height, fps):
    global running, image_f
    image_f = cv2.imread(file_name)
    running = True

class OwnImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.canny.stateChanged.connect(self.update_setting)
        self.sobel.stateChanged.connect(self.update_setting)
        self.sobel_x.stateChanged.connect(self.update_setting)
        self.sobel_y.stateChanged.connect(self.update_setting)
        self.laplacian.stateChanged.connect(self.update_setting)
        self.inversion.stateChanged.connect(self.update_setting)
        self.hist_eq.stateChanged.connect(self.update_setting)
        self.gray.stateChanged.connect(self.update_setting)
        self.corner.stateChanged.connect(self.update_setting)
        self.blob.stateChanged.connect(self.update_setting)
        self.canny_max.valueChanged.connect(self.update_setting)
        self.canny_min.valueChanged.connect(self.update_setting)
        self.brightness_value.valueChanged.connect(self.update_setting)
        self.contrast_value.valueChanged.connect(self.update_setting)
        self.erosion_value.valueChanged.connect(self.update_setting)
        self.dilation_value.valueChanged.connect(self.update_setting)
        self.corner_maxc.valueChanged.connect(self.update_setting)
        self.corner_mind.valueChanged.connect(self.update_setting)
        self.corner_quality.valueChanged.connect(self.update_setting)
        self.rotate.valueChanged.connect(self.update_setting)
        self.translate_x.valueChanged.connect(self.update_setting)
        self.translate_y.valueChanged.connect(self.update_setting)
        self.zoom_value.valueChanged.connect(self.update_setting)
        self.gaussian_blur.valueChanged.connect(self.update_setting)
        self.median_blur.valueChanged.connect(self.update_setting)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)       
        self.ImgWidget_O = OwnImageWidget(self.ImgWidget_O)       
        self.update_frame()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)
        
        self.startWebcam.setEnabled(False)
        self.startVideo.setEnabled(False)
        self.startImage.clicked.connect(self.change_image)
        self.dft.clicked.connect(self.show_dft)
        
    def show_dft(self):
        f = np.fft.fft2(image_f)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('TU Project\n--------------------------\nDiscrete Fourier Transform')
        plt.show()
        
    def change_image(self):
        global image_f, running
        file_name = easygui.fileopenbox(filetypes = ["*.png","*.jpg"])
        image_f = cv2.imread(file_name)
        running = True
        
    def update_setting(self):
        global running
        running = True
        
    def update_frame(self):
        global running, image, dft
        if running:
            img = image_f.copy()
            
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1
            
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
            height, width, bpc = img.shape
            bpl = bpc * width
            img_o = img.copy()     

            if self.inversion.isChecked():
                img = cv2.bitwise_not(img)

            if self.brightness_value.value() > 0:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                v += self.brightness_value.value()
                #v = np.where((255 - v) < 255,255,v+self.brightness_value.value())
                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)

            if self.zoom_value.value() != 100:
                resize_val = int(float(self.zoom_value.value())*0.1)
                print resize_val
                img = cv2.resize(img,None,fx=resize_val, fy=resize_val)

            if self.translate_x.value() != 100:
                M = np.float32([[1,0,self.translate_x.value()-100],[0,1,0]])
                rows,cols,d = img.shape
                img = cv2.warpAffine(img,M,(cols,rows))

            if self.translate_y.value() != 100:
                M = np.float32([[1,0,0],[0,1,self.translate_y.value()-100]])
                rows,cols,d = img.shape
                img = cv2.warpAffine(img,M,(cols,rows))

            if self.hist_eq.isChecked():
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.equalizeHist(img)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

            if self.gray.isChecked():
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

            if self.gaussian_blur.value()>3:
                img = cv2.GaussianBlur(img,(self.gaussian_blur.value(),self.gaussian_blur.value()),0)

            if self.median_blur.value()>3:
                img = cv2.medianBlur(img,self.median_blur.value())

            if self.rotate.value() > 0:
                cols = img.shape[1]
                rows = img.shape[0]
                M = cv2.getRotationMatrix2D((cols/2,rows/2),360-self.rotate.value(),1)
                img = cv2.warpAffine(img,M,(cols,rows))

            kernel = np.ones((5,5),np.uint8)
            if self.erosion_value.value() > 0:
                img = cv2.erode(img,kernel,iterations = self.erosion_value.value())
            if self.dilation_value.value() > 0:
                img = cv2.dilate(img,kernel,iterations = self.dilation_value.value())
    
            if self.canny.isChecked():
                img =  cv2.Canny(img,self.canny_min.value(),self.canny_max.value())
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            elif self.laplacian.isChecked():
                img = cv2.Laplacian(img,cv2.CV_8U)
            elif self.sobel.isChecked():
                if self.sobel_x.isChecked():
                    img = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
                if self.sobel_y.isChecked():
                    img = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
            
            if self.corner.isChecked():
                gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                gray = np.float32(gray)
                corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
                corners=cv2.goodFeaturesToTrack(gray,
                                                 self.corner_maxc.value(),
                                                 0.0001*float(self.corner_quality.value()),
                                                 self.corner_mind.value())
                corners=np.int0(corners)
                for corner in corners:
                    x,y = corner.ravel()
                    cv2.circle(img,(x,y),3,255,-1)   
                    
            if self.blob.isChecked():
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                lower_blue = np.array([110,50,50])
                upper_blue = np.array([130,255,255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                img = cv2.bitwise_and(img,img,mask= mask)

            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            image_o = QtGui.QImage(img_o.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            
            self.ImgWidget.setImage(image)
            self.ImgWidget_O.setImage(image_o)
            running = False

    def closeEvent(self, event):
        global running
        running = False


file_name = easygui.fileopenbox(filetypes = ["*.png","*.jpg"])
capture_thread = threading.Thread(target=grab, args = (file_name, q, 1920, 1080, 15))
capture_thread.start()
app = QtGui.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('TU Image Processing Project')
w.show()
app.exec_()
