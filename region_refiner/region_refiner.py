import cv2 as cv
import numpy as np

from connected_components.connected_components import get_connected_components


class RegionRefiner:

    def __init__(self, img_shape, ccs_text, ccs_non_text):
        self.__img_shape = img_shape
        self.__ccs_text = ccs_text.copy()
        self.__ccs_non_text = ccs_non_text.copy()

    def remove_intersected_regions(self):
        img_text = np.zeros(self.__img_shape, np.uint8)
        img_non_text = np.zeros(self.__img_shape, np.uint8)
        cv.drawContours(img_text, [cc.get_contour() for cc in self.__ccs_text], -1, 255, -1)
        cv.drawContours(img_non_text, [cc.get_contour() for cc in self.__ccs_non_text], -1, 255, -1)
        ccs_non_text = get_connected_components(img_non_text, external=True)
        ccs_text_new = []
        for cc_non_text in ccs_non_text.copy():
            for cc_text in self.__ccs_text:
                # if cc_text.contains(cc_non_text) or does_intersect(self.__img_shape, cc_text, cc_non_text):
                # if includes(self.__img_shape, cc_text, cc_non_text):
                if cc_text.contains(cc_non_text):
                    x, y, w, h = cc_non_text.get_rect()
                    cv.rectangle(img_text, (x, y), (x + w, y + h), 255, -1)
                    ccs_text_new.append(cc_non_text)
                    ccs_non_text.remove(cc_non_text)
                    break

        cv.namedWindow('IMg text1', cv.WINDOW_FREERATIO)
        cv.imshow('IMg text1', img_text)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
        kernel = np.ones((5, 1), np.uint8)
        img_text = cv.morphologyEx(img_text, cv.MORPH_CLOSE, kernel, iterations=4)
        cv.namedWindow('IMg text2', cv.WINDOW_FREERATIO)
        cv.imshow('IMg text2', img_text)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
        self.__ccs_text = get_connected_components(img_text, external=True)
        self.__ccs_non_text = ccs_non_text
        return self.__ccs_text.copy(), self.__ccs_non_text.copy(), ccs_text_new
