import cv2
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tensorflow import keras
from core import locate_and_correct
from Unet import unet_predict
from CNN import cnn_predict

unet = keras.models.load_model('unet.h5')
cnn = keras.models.load_model('cnn.h5')
print('正在启动中,请稍等...')
cnn_predict(cnn, [np.zeros((80, 240, 3))])