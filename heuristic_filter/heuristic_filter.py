import cv2 as cv
import numpy as np

from connected_components.connected_components import get_connected_components

T_AREA = 6
T_INSIDE = 4
T_DENS = 0.06
T_RATE = 0.06


class HeuristicFilter:
    def __init__(self, img, t_area=T_AREA, t_inside=T_INSIDE, t_dens=T_DENS, t_rate=T_RATE):
        super().__init__()
        self.__img = img.copy()
        self.__t_inside = t_inside
        self.__t_area = t_area
        self.__t_dens = t_dens
        self.__t_rate = t_rate

    def filter(self):
        """
        Heuristic filter.

        Filters out not-text connected components (CCs) from the input ccs.

        Returns:
            ccs_text: CCs of text components
            ccs_img: CCs of non_text components
        """

        ccs_noise = self.__get_ccs_noise()

        ccs_denoise = self.__filter_noise(ccs_noise)

        ccs_text, ccs_non_text = self.__filter_non_text(ccs_denoise)

        ccs_non_text.extend(ccs_noise)

        return ccs_text, ccs_non_text, self.__img

    def __get_ccs(self):
        return get_connected_components(self.__img)

    def __get_ccs_noise(self):
        ccs = self.__get_ccs()
        ccs_noise = []
        for cc in ccs:
            if cc.get_area() < self.__t_area or cc.get_dens() < self.__t_dens or cc.get_hw_rate() < self.__t_rate:
                # if cc.get_area() < self.__t_area or cc.get_dens() < self.__t_dens:
                ccs_noise.append(cc)

        return ccs_noise

    def __filter_noise(self, ccs_noise):
        contours = [cc.get_contour() for cc in ccs_noise]
        cv.drawContours(self.__img, contours, -1, (0, 0, 0), -1)
        ccs = self.__get_ccs()
        return ccs

    def __filter_non_text(self, ccs):
        ccs_non_text = []
        for i, cc in enumerate(ccs):
            if self.__has_descendants_more_than_t_inside(cc, ccs):
                ccs_non_text.append(cc)
                cv.drawContours(
                    self.__img, [cc.get_contour()], -1, (0, 0, 0), -1)

        kernel = np.ones((3, 3), np.uint8)
        # self.__img = cv.morphologyEx(self.__img, cv.MORPH_CLOSE, kernel)

        ccs_text = self.__get_ccs()

        return ccs_text, ccs_non_text

    def __has_descendants_more_than_t_inside(self, cc, ccs):
        num_descendants = 0
        for cc_j in ccs:
            if cc.contains(cc_j):
                num_descendants += 1
                if num_descendants > self.__t_inside + 1:
                    return True

        return False
