from typing import List

import cv2 as cv
import numpy as np

from connected_components.connected_components import ConnectedComponent
from mll_classifier.recursive_filter import RecursiveFilter
from mll_classifier.region import Region

T_VAR = 1.3


class MllClassifier:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()
        self.__region = Region((0, 0, self.__img.shape[1], self.__img.shape[0]), self.__img)

    def classify_non_text_ccs(self):
        ccs_non_text = self.__apply_recursive_filter()
        return self.__region.get_ccs(), ccs_non_text, self.__img

    def __apply_recursive_filter(self):
        non_text = self.__apply_multilevel_classification()
        non_text2 = self.apply_multilayer_classification()
        non_text.extend(non_text2)
        return non_text

    def __apply_multilevel_classification(self) -> List[ConnectedComponent]:
        non_text = []
        regions = [self.__region]
        sum_u = np.sum(regions[0].get_img())
        sum_v = 0
        while len(regions) > 0:
            regions = self.__get_homogeneous_regions(regions)

            for region in regions:
                modified, _, non_text2 = RecursiveFilter(region).filter()
                non_text.extend(non_text2)

                if not modified:
                    regions.remove(region)
                else:
                    sum_v_i = np.sum(region.get_img())
                    if sum_v_i == 0:
                        regions.remove(region)
                    else:
                        sum_v += sum_v_i

            if sum_v == 0 or sum_v / sum_u >= 0.9:
                break
            sum_u = sum_v
            sum_v = 0

        cv.drawContours(self.__img, [cc.get_contour() for cc in non_text], -1, 0, -1)

        return non_text

    def apply_multilayer_classification(self) -> List[ConnectedComponent]:
        non_text = []
        modified = True
        while modified:
            modified = False
            regions = [
                Region((0, 0, self.__img.shape[1],
                        self.__img.shape[0]), self.__img)
            ]
            sum_u = np.sum(regions[0].get_img())
            sum_v = 0
            regions = self.__get_homogeneous_regions(regions)

            for region in regions:
                region_modified, _, non_text2 = RecursiveFilter(region).filter()
                non_text.extend(non_text2)
                cv.drawContours(self.__img, [cc.get_contour() for cc in non_text2], -1, 0, -1)

                modified = modified or region_modified
                sum_v += np.sum(region.get_img())

            if sum_v / sum_u >= 0.9:
                break

        self.__region = Region((0, 0, self.__img.shape[1], self.__img.shape[0]), self.__img)

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
        return self.__get_homogeneous_regions([self.__region])

    def get_highest_level_homogeneous_regions(self):
        regions = [self.__region]
        finished_regions = []
        while len(regions) > 0:
            region_changed, next_regions = regions[0].get_next_level_homogeneous_regions(
            )

            if region_changed:
                regions.extend(next_regions)
                regions.pop(0)
            else:
                finished_regions.extend(next_regions)
                regions.pop(0)

        return finished_regions

    def get_region(self) -> Region:
        return self.__region
