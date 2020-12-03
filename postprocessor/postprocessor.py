import cv2 as cv
import numpy as np


class Postprocessor:
    def __init__(self, img, ccs_text, ccs_non_text):
        super().__init__()
        self.__img = img.copy()
        self.__ccs_text = ccs_text.copy()
        self.__ccs_non_text = ccs_non_text.copy()

    def postprocess(self):
        self.__remove_intersected()
        img_text = cv.drawContours(
            self.__img.copy(), [cc.get_contour() for cc in self.__ccs_non_text], -1, 0, -1)
        return self.__ccs_text, self.__ccs_non_text, img_text

    def __remove_intersected(self):
        non_text_img = self.__get_non_text_image()
        non_text_img = self.__dilate(non_text_img)
        for i, cc in enumerate(self.__ccs_text):
            if self.__cc_intersect_with_non_text_img(cc, non_text_img):
                self.__ccs_non_text.append(self.__ccs_text.pop(i))

    def __get_non_text_image(self):
        non_text_image = self.__img.copy()
        for cc in self.__ccs_text:
            x, y, w, h = cc.get_rect()
            cv.rectangle(non_text_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

        return non_text_image

    def __dilate(self, img):
        x_size = int((img.shape[0] * 0.5) / 100)
        y_size = int((img.shape[1] * 0.5) / 100)
        kernel = np.ones((x_size, y_size), np.uint8)
        return cv.dilate(img, kernel, iterations=1)

    def __cc_intersect_with_non_text_img(self, cc, non_text_img):
        blank = np.zeros(self.__img.shape[0:2])

        x, y, w, h = cc.get_rect()
        img = cv.rectangle(blank.copy(), (x, y), (x + w, y + h), 255)

        intersection = np.logical_and(img, non_text_img)

        return intersection.any()
