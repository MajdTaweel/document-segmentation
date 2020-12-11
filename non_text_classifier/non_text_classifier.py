import cv2 as cv
import numpy as np

from connected_components.connected_components import ConnectedComponent
from mll_classifier.mll_classifier import MllClassifier


class NonTextClassifier:

    def __init__(self, img_shape, ccs_text, ccs_non_text):
        super().__init__()
        self.__img_shape = img_shape
        self.__ccs_text = ccs_text.copy()
        self.__ccs_non_text = ccs_non_text.copy()
        self.__ccs_negative_text = []
        self.__h_lines = []
        self.__v_lines = []
        self.__tables = []
        self.__separators = []
        self.__graphics = []

    def classify_non_text_elements(self):
        for cc in self.__ccs_non_text.copy():
            self.__classify_cc(cc)

        return {
            'Paragraph': self.__ccs_text,
            'Negative Text': self.__ccs_negative_text,
            'H Line': self.__h_lines,
            'V Line': self.__v_lines,
            'Table': self.__tables,
            'Separator': self.__separators,
            'Image': self.__graphics
        }

    def __classify_cc(self, cc: ConnectedComponent):
        if cc.get_area() <= 50:
            self.__ccs_non_text.remove(cc)
        elif self.__is_negative_text(cc):
            self.__ccs_negative_text.append(cc)
            # self.__ccs_text.append(cc)
            self.__ccs_non_text.remove(cc)
        elif self.__is_line(cc):
            if self.__is_h_line(cc):
                self.__h_lines.append(cc)
            else:
                self.__v_lines.append(cc)
            self.__ccs_non_text.remove(cc)
        elif self.__is_table(cc):
            self.__tables.append(cc)
            self.__ccs_non_text.remove(cc)
        elif self.__is_separator(cc):
            self.__separators.append(cc)
            self.__ccs_non_text.remove(cc)
        else:
            self.__graphics.append(cc)
            self.__ccs_non_text.remove(cc)

    def __is_negative_text(self, cc: ConnectedComponent):
        if cc.get_dens() < 0.9:
            return False

        x, y, w, h = cc.get_rect()
        blank = np.zeros((h, w), np.uint8)
        negative_candidate = cv.drawContours(blank, [cc.get_contour()], -1, 255, -1)
        negative_candidate = cv.bitwise_not(negative_candidate)
        # blank = np.ones((h, w), np.uint8)
        # negative_candidate = cv.drawContours(blank, [cc.get_contour()], -1, 0, -1)

        ccs_text, ccs_non_text = MllClassifier(negative_candidate).apply_multilayer_classification()

        area_text = 0
        for cc_text in ccs_text:
            area_text += cc_text.get_area()

        area_non_text = 0
        for cc_non_text in ccs_non_text:
            area_non_text += cc_non_text.get_area()

        return area_text > area_non_text

    def __is_line(self, cc):
        return cc.get_hw_rate() <= 0.1

    def __is_h_line(self, cc):
        _, _, w, h = cc.get_rect()
        return w > h

    def __is_table(self, cc):
        return False

    def __is_separator(self, cc):
        if cc.get_dens() > 0.02:
            return False
        blank = np.zeros(self.__img_shape, np.uint8)

        x, y, w, h = cc.get_rect()
        img1 = cv.rectangle(blank.copy(), (x, y), (x + w, y + h), 255, -1)
        img2 = cv.drawContours(blank.copy(), [cc2.get_contour() for cc2 in self.__ccs_text], -1, 255, -1)

        ccs_intersection = np.logical_and(img1, img2)

        return ccs_intersection.any()
