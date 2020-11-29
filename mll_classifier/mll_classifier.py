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
        non_text = []

        # Multilevel classification
        modified = True
        regions = self.__regions
        sum_u = np.sum(regions[0].get_img())
        sum_v = 0
        while len(regions) > 0:
            regions = self.__get_homogeneous_regions(regions)

            for region in regions:
                modified, self.__img, non_text2 = RecursiveFilter(
                    self.__img, region).filter()
                non_text.extend(non_text2)

                if not modified:
                    regions.remove(region)
                else:
                    sum_v_i = np.sum(region.get_img())
                    if sum_v_i == 0:
                        regions.remove(region)
                    else:
                        sum_v += np.sum(sum_v_i)

            if sum_v == 0 or sum_v / sum_u >= 0.9:
                break
            sum_u = sum_v
            sum_v = 0

        # Multi-layer classification
        regions = [
            Region((0, 0, self.__img.shape[1],
                    self.__img.shape[0]), self.__img)
        ]
        sum_u = np.sum(regions[0].get_img())
        modified = True
        while modified:
            sum_v = 0
            modified = False
            regions = self.__get_homogeneous_regions(regions)

            for region in regions:
                region_modified, self.__img, non_text2 = RecursiveFilter(
                    self.__img, region).filter()
                non_text.extend(non_text2)

                modified = modified or region_modified
                sum_v += np.sum(region.get_img())

            if sum_v / sum_u >= 0.9:
                break
            sum_u = sum_v
            sum_v = 0

        return non_text

    def __get_homogeneous_regions(self, regions):
        i = 0
        new_regions = []
        while i < len(regions):
            region_changed, next_regions = regions[i].get_next_level_homogeneous_regions(
            )

            new_regions.extend(next_regions)

            i += 1
        return new_regions

    def get_next_level_homogeneous_regions(self):
        return self.__get_homogeneous_regions(self.__regions)
        # regions = self.__regions
        # i = 0
        # finished_regions = []
        # while i < len(regions):
        #     region_changed, next_regions = regions[i].get_next_level_homogeneous_regions(
        #     )

        #     if region_changed:
        #         regions.extend(next_regions)
        #         regions.pop(i)
        #     else:
        #         finished_regions.extend(next_regions)
        #         i += 1

        # return finished_regions
