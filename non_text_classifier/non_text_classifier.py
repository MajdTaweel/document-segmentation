import cv2 as cv
import numpy as np

from connected_components.connected_components import ConnectedComponent
from mll_classifier.mll_classifier import MllClassifier


class NonTextClassifier:

    def __init__(self, non_text_img, ccs_text, ccs_non_text):
        super().__init__()
        self.__non_text_img = non_text_img.copy()
        self.__ccs_text = ccs_text.copy()
        self.__ccs_non_text = ccs_non_text.copy()
        self.__ccs_negative_text = []
        self.__h_lines = []
        self.__v_lines = []
        self.__tables = []
        self.__separators = []
        self.__graphics = []

    def __classify_cc(self, cc: ConnectedComponent):
        if self.__is_negative_text(cc):
            # self.__ccs_negative_text.append(cc)
            self.__ccs_text.append(cc)
            self.__ccs_non_text.remove(cc)
        elif self.__is_line(cc):
            if self.__is_h_line(cc):
                self.__h_lines.append(cc)
            else:
                self.__v_lines.append(cc)
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
        negative_candidate = cv.drawContours(blank, [cc], -1, 255, -1)
        negative_candidate = cv.bitwise_not(negative_candidate)

        mll = MllClassifier(negative_candidate)

        ccs_non_text = mll.apply_multilayer_classification()
        region = mll.get_region()

        ccs_text = region.get_ccs()
        area_text = 0

        for cc_text in ccs_text:
            area_text += cc_text.get_area()
        area_non_text = 0

        for cc_non_text in ccs_non_text:
            area_non_text += cc_non_text.get_area()

        return area_text > area_non_text

    def __is_line(self, cc):
        return cc.get_dens() <= 0.1

    def __is_h_line(self, cc):
        _, _, w, h = cc.get_rect()
        return w > h

    def __is_table(self, cc):
        return False

    def __is_separator(self, cc):
        return False
