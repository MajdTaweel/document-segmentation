import cv2 as cv


class Binarizer:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()

    def binarize(self):
        ret, self.__img = cv.threshold(self.__img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        if ret < 100:
            self.__img = cv.bitwise_not(self.__img)

        return self.__img
