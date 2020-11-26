import cv2 as cv
import numpy as np
from connected_components.connected_components import get_connected_components
from matplotlib import pyplot as plt

from mll_classifier.recursive_filter import RecursiveFilter
from mll_classifier.region import Region

T_VAR = 1.3


class MllClassifier:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()
        self.__regions = [
            Region(
                (0, 0, self.__img.shape[1], self.__img.shape[0]),
                self.__img
            )
        ]

    def classify_non_text_ccs(self):
        ccs_non_text = self.__apply_recursive_filter()
        ccs_text = get_connected_components(self.__img)
        return ccs_text, ccs_non_text, self.__img

    def __apply_recursive_filter(self):
        modified = True
        non_text = []
        while modified:
            self.__regions = self.__get_homogeneous_regions(self.__regions)
            modified, self.__img, non_text2 = RecursiveFilter(
                self.__img, self.__regions).filter()
            non_text.extend(non_text2)

        return non_text

    def __get_homogeneous_regions(self, regions):
        i = 0
        while i < len(regions):
            region_changed, next_regions = regions[i].set_img(
                self.__img).get_next_level_homogeneous_regions()
            if region_changed:
                regions.extend(next_regions)
                regions.pop(i)
            else:
                i += 1
        return regions
