import cv2 as cv
# import numpy as np
# from skimage.filters import threshold_sauvola
# from skimage.util import img_as_ubyte


class Binarizer:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()

    def binarize(self):
        ret, self.__img = cv.threshold(self.__img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        if ret < 100:
            self.__img = cv.bitwise_not(self.__img)

        return self.__img

        # # img = img_as_float(self.__img)
        # img = self.__img.copy()
        # window_size = 25
        # thresh_sauvola = threshold_sauvola(img,  window_size=window_size, r=128)
        # binary_sauvola = img <= thresh_sauvola
        # binary_sauvola = img_as_ubyte(binary_sauvola)
        #
        # return binary_sauvola
