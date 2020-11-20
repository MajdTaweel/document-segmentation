import cv2 as cv


class Binarizer:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()

    def binarize(self):
        # self.__img = cv.adaptiveThreshold(
        #     self.__img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, -2)
        _, self.__img = cv.threshold(
            self.__img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return self.__img
