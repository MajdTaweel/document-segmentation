import cv2 as cv
import numpy as np
import util.img as iu

from connected_components.connected_components import get_connected_components, intersection_percentage, intersection


class RegionRefiner:

    def __init__(self, debug=False):
        self.__colors = {
            'Paragraph': ((73, 48, 0), (255, 255, 255)),
            'Negative Text': ((40, 40, 214), (255, 255, 255)),
            'H Line': ((0, 127, 247), (255, 255, 255)),
            'V Line': ((73, 191, 252), (0, 0, 0)),
            'Table': ((183, 226, 234), (0, 0, 0)),
            'Separator': ((36, 0, 71), (255, 255, 255)),
            'Image': ((199, 136, 86), (255, 255, 255))
        }
        self.__debug = debug

    def remove_intersected_regions(self, img_text, ccs_non_text):
        img_text = img_text.copy()
        ccs_non_text = ccs_non_text.copy()
        kernel = np.ones((3, 1), np.uint8)
        # dilation = cv.dilate(img_text, kernel)
        closing = cv.morphologyEx(img_text, cv.MORPH_CLOSE, kernel, iterations=4)
        ccs_text = get_connected_components(closing, external=True)
        img_text = np.zeros(img_text.shape, np.uint8)
        img_non_text = np.zeros(img_text.shape, np.uint8)
        cv.drawContours(img_text, [cc.get_contour() for cc in ccs_text], -1, 255, -1)
        cv.drawContours(img_non_text, [cc.get_contour() for cc in ccs_non_text], -1, 255, -1)
        ccs_non_text = get_connected_components(img_non_text, external=True)
        ccs_text_new = []
        for cc_non_text in ccs_non_text.copy():
            for cc_text in ccs_text:
                # if cc_text.contains(cc_non_text) or does_intersect(self.__img_shape, cc_text, cc_non_text):
                # if includes(img_shape, cc_text, cc_non_text):
                # if cc_text.contains(cc_non_text):
                if cc_text.get_rect_area() > cc_non_text.get_rect_area() and \
                        intersection_percentage(cc_text, cc_non_text) >= 0.9:
                    x, y, w, h = cc_non_text.get_rect()
                    cv.rectangle(img_text, (x, y), (x + w, y + h), 255, -1)
                    ccs_text_new.append(cc_non_text)
                    ccs_non_text.remove(cc_non_text)
                    break

        ccs_non_text_new = []
        for cc_text in ccs_text:
            for cc_non_text in ccs_non_text:
                # if cc_non_text.contains(cc_text) and cc_non_text.get_dens() > 0.02:
                if cc_non_text.get_rect_area() > cc_text.get_rect_area() and \
                        intersection_percentage(cc_non_text, cc_text) >= 0.9 and cc_non_text.get_dens() > 0.02:
                    x, y, w, h = cc_text.get_rect()
                    cv.rectangle(img_text, (x, y), (x + w, y + h), 0, -1)
                    ccs_non_text_new.append(cc_text)
                    break

        ccs_non_text.extend(ccs_non_text_new)

        if self.__debug:
            iu.show_and_wait('Intersections Grouping', img_text)
        kernel = np.ones((3, 3), np.uint8)
        img_text = cv.morphologyEx(img_text, cv.MORPH_CLOSE, kernel, iterations=4)
        if self.__debug:
            iu.show_and_wait('Intersections Grouping (Morph-Close)', img_text)
        ccs_text = get_connected_components(img_text, external=True)
        return ccs_text, ccs_non_text, ccs_text_new, ccs_non_text_new

    def refine_non_text_regions(self, img_shape, ccs_non_text):
        rect_ccs = []
        img_non_text = np.zeros(img_shape, np.uint8)
        img_rect_non_text = np.zeros(img_shape, np.uint8)
        cv.drawContours(img_non_text, [cc.get_contour() for cc in ccs_non_text], -1, 255, -1)
        for cc1 in ccs_non_text:
            if cc1.get_dens() <= 0.02:
                continue
            intersections_sum = 0
            areas_sum = 0
            for cc2 in ccs_non_text:
                if cc1 is cc2:
                    continue
                rects_intersection = intersection(cc1.get_rect(), cc2.get_rect())
                if len(rects_intersection) > 0:
                    areas_sum += cc2.get_rect_area()
                    intersections_sum += rects_intersection[2] * rects_intersection[3]
            if areas_sum > 0 and intersections_sum / areas_sum >= 0.9:
                x, y, w, h = cc1.get_rect()
                cv.rectangle(img_non_text, (x, y), (x + w, y + h), 0, -1)
                cv.rectangle(img_rect_non_text, (x, y), (x + w, y + h), 255, -1)
                rect_ccs.append(cc1)
        return get_connected_components(img_non_text), get_connected_components(img_rect_non_text, external=True)

    def label_regions(self, img, ccs):
        img = img.copy()
        contoured_regions = img.copy()
        labeled_regions = img.copy()
        for key in ccs.keys():
            for cc in ccs[key]:
                cv.drawContours(contoured_regions, [cc.get_contour()], -1, self.__colors[key][0], 2)
                cv.drawContours(labeled_regions, [cc.get_contour()], -1, self.__colors[key][0], 2)
                x, y, _, _ = cc.get_rect()
                ((w, h), b) = cv.getTextSize(key, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)
                cv.rectangle(labeled_regions, (x + 2, y + 2), (x + w + 4, y + h + 4), self.__colors[key][0], -1)
                cv.putText(labeled_regions, key, (x + 4, y + b + 4), cv.FONT_HERSHEY_DUPLEX, 0.5, self.__colors[key][1], 1)
        return cv.addWeighted(contoured_regions, 0.75, img, 0.25, 0), cv.addWeighted(labeled_regions, 0.5, img, 0.5, 0)
