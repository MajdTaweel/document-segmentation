import cv2 as cv
import numpy as np

from connected_components.connected_components import get_connected_components


class RegionRefiner:

    def __init__(self):
        self.__colors = {
            'Paragraph': ((73, 48, 0), (255, 255, 255)),
            'Header': ((40, 40, 214), (255, 255, 255)),
            'H Lines': ((0, 127, 247), (255, 255, 255)),
            'V Lines': ((73, 191, 7252), (0, 0, 0)),
            'Table': ((183, 226, 234), (0, 0, 0)),
            'Separator': ((36, 0, 71), (255, 255, 255)),
            'Image': ((199, 136, 86), (255, 255, 255))
        }

    @staticmethod
    def remove_intersected_regions(img_shape, ccs_text, ccs_non_text):
        img_text = np.zeros(img_shape, np.uint8)
        img_non_text = np.zeros(img_shape, np.uint8)
        cv.drawContours(img_text, [cc.get_contour() for cc in ccs_text], -1, 255, -1)
        cv.drawContours(img_non_text, [cc.get_contour() for cc in ccs_non_text], -1, 255, -1)
        ccs_non_text = get_connected_components(img_non_text, external=True)
        ccs_text_new = []
        for cc_non_text in ccs_non_text.copy():
            for cc_text in ccs_text:
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
        ccs_text = get_connected_components(img_text, external=True)
        return ccs_text.copy(), ccs_non_text.copy(), ccs_text_new

    def label_regions(self, img, ccs):
        img = img.copy()
        for key in ccs.keys():
            for cc in ccs[key]:
                cv.drawContours(img, [cc.get_contour()], -1, self.__colors[key][0], 2)
                x, y, _, _ = cc.get_rect()
                ((w, h), b) = cv.getTextSize(key, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv.rectangle(img, (x + 2, y + 2), (x + w + 4, y + h + 4), self.__colors[key][0], -1)
                cv.putText(img, key, (x + 4, y + b + 4), cv.FONT_HERSHEY_DUPLEX, 0.5, self.__colors[key][1], 1)
        return img
