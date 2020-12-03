import cv2 as cv
import numpy as np

from connected_components.connected_components import get_connected_components, union
from mll_classifier.mll_classifier import MllClassifier
from mll_classifier.region import Region
from text_segmenter.white_space_filter import WhiteSpaceFilter


class TextSegmenter:
    def __init__(self, img, ccs_text, src):
        super().__init__()
        self.__img = img.copy()
        self.__ccs_text = ccs_text
        self.src = src.copy()

    def segment_text(self):
        self.__filter_ws()
        hrs = self.__segment_paragraphs()
        ccs_text = self.__smooth_regions(hrs)

        cv.namedWindow('Bounding Box Smoothed', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box Smoothed', self.__img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

        return ccs_text

    def __filter_ws(self):
        self.__img = WhiteSpaceFilter(self.__img, self.src).filter_ws()

    def __segment_paragraphs(self):
        hrs = MllClassifier(
            self.__img).get_next_level_homogeneous_regions()

        new_hrs = []
        for hr in hrs:
            next_hrs = self.__segment_region_paragraphs(hr)
            new_hrs.extend(next_hrs)

        return new_hrs

    def __segment_region_paragraphs(self, hr):
        hcs = hr.get_hcs()

        if len(hcs) <= 3:
            return [hr]

        page_width = self.__img.shape[1]

        block_width = 0
        for hc in hcs:
            chain_width = (hc[-1].get_rect()[0] +
                           hc[-1].get_rect()[2]) - hc[0].get_rect()[0]
            block_width = np.max([block_width, chain_width])

        if block_width < page_width / 8:
            return [hr]

        lines = self.__get_paragraph_lines(hcs)

        split_hrs = self.__segment_paragraph(hr, lines)

        return split_hrs

    @staticmethod
    def __segment_paragraph(hr, lines):
        prev = 0
        curr = 1
        nextt = 2

        th = 0.8

        split_lines = []
        while nextt < len(lines):
            x_p, y_p, w_p, h_p = lines[prev]
            x_c, y_c, w_c, h_c = lines[curr]
            x_n, y_n, w_n, h_n = lines[nextt]
            if w_c <= w_p * th:
                if w_n <= w_p * th:
                    split_lines.append(union(lines[curr], lines[nextt]))
                elif w_c <= w_n * th and (x_c * th > x_n or x_c + w_c < x_n + w_n * th):
                    split_lines.append(union(lines[prev], lines[curr]))
            prev += 1
            curr += 1
            nextt += 1

        splits = [rect[1] + int(rect[3] / 2) for rect in split_lines]

        split_hrs = hr.split_horizontally_at(splits)

        return split_hrs

    def __get_paragraph_lines(self, hcs):
        lines = []

        for hc in hcs:
            line = None
            for cc in hc:
                if line is None:
                    line = cc.get_rect()
                    continue
                line = union(line, cc.get_rect())
                x, y, w, h = line
                self.__img = cv.rectangle(self.__img, (x, int(y + h / 4)), (x + w, int(y + h / 2)), 255, -1)
            lines.append(line)

        return lines

    def __smooth_regions(self, hrs):
        ccs_text = []
        for hr in hrs:
            ccs = self.__smooth_region(hr)
            ccs_text.extend(ccs)

        blank = np.zeros(self.__img.shape[:2], np.uint8)
        self.__img = cv.drawContours(blank, [cc.get_contour() for cc in ccs_text], -1, 255, -1)

        return ccs_text

    @staticmethod
    def __smooth_region(hr: Region):
        img = hr.get_img()

        kernel_height = 2
        wl = [wl_i[3] for wl_i in hr.get_wl_h()]
        if len(wl) > 0:
            kernel_height = round(2 * np.percentile(wl, 75))

        kernel_width = 2
        ws = hr.set_features().get_features()['ws']
        if len(ws) > 0:
            kernel_width = round(2 * np.percentile(ws, 75))

        kernel = np.ones((kernel_height, kernel_width), np.uint8)

        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=2)

        # kernel = np.ones((kernel_height * 2, kernel_width * 2), np.uint8)
        #
        # dilation = cv.dilate(closing, kernel)
        # erosion = cv.erode(dilation, kernel)

        return get_connected_components(closing, (hr.get_rect()[0], hr.get_rect()[1]))
