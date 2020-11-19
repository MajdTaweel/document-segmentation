import cv2 as cv
from binarizer import Binarizer


KERNEL_SIZE = 5


class Preprocessor:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()

    def preprocess(self):
        self.__img_to_gray_scale()
        # self.__enhance_img()
        # self.__smooth_img()
        binarizer = Binarizer(self.__img)
        self.__img = binarizer.binarize()
        self.__correct_skew()
        return self.__img

    def __img_to_gray_scale(self):
        if len(self.__img.shape) == 3:
            self.__img = cv.cvtColor(self.__img, cv.COLOR_BGR2GRAY)

    def __enhance_img(self):
        self.__img = cv.equalizeHist(self.__img)

    def __smooth_img(self):
        # self.__img = cv.GaussianBlur(self.__img, (KERNEL_SIZE, KERNEL_SIZE), 0)
        self.__img = cv.medianBlur(self.__img, KERNEL_SIZE)

    def __correct_skew(self):
        pts = cv.findNonZero(self.__img)
        ret = cv.minAreaRect(pts)

        (cx, cy), (w, h), ang = ret
        if w > h:
            w, h = h, w
            ang += 90

        M = cv.getRotationMatrix2D((cx, cy), ang, 1.0)
        self.__img = cv.warpAffine(
            self.__img, M, (self.__img.shape[1], self.__img.shape[0]))
