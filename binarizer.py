import cv2 as cv


class Binarizer:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()

    def binarize(self):
        self.__img_to_gray_scale()
        return cv.bitwise_not(self.__binarize())

    def __binarize(self):
        self.__img = cv.adaptiveThreshold(
            self.__img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 199, 5)
        return self.__img

    def __img_to_gray_scale(self):
        if len(self.__img.shape) == 3:
            self.__img = cv.cvtColor(self.__img, cv.COLOR_BGR2GRAY)
