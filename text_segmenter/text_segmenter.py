import cv2 as cv
import numpy as np
import util.img as iu

from connected_components.connected_components import get_connected_components, union
from mll_classifier.mll_classifier import MllClassifier
from mll_classifier.region import Region
from region_refiner.region_refiner import RegionRefiner
from text_segmenter.white_space_filter import WhiteSpaceFilter

THETA = 1.2


class TextSegmenter:
    def __init__(self, img, ccs_text, ccs_non_text, src, debug=False):
        super().__init__()
        self.__img = img.copy()
        self.__ccs_text = ccs_text.copy()
        self.__ccs_non_text = ccs_non_text.copy()
        self.src = src.copy()
        self.__debug = debug

    def segment_text(self):
        self.__filter_ws()
        text_blocks = self.__get_text_blocks()
        hrs = self.__segment_paragraphs(text_blocks)
        ccs_text = self.__smooth_regions(hrs)
        return ccs_text, self.__ccs_non_text

    def __get_text_blocks(self):
        ccs, self.__ccs_non_text, ccs_text_new, ccs_non_text_new = RegionRefiner(
            self.__debug).remove_intersected_regions(self.__img, self.__ccs_non_text)
        for cc_text_new in ccs_text_new:
            x, y, w, h = cc_text_new.get_rect()
            cv.rectangle(self.__img, (x, y + 3), (x + w, y + h - 3), 255, -1)

        for cc_non_text_new in ccs_non_text_new:
            x, y, w, h = cc_non_text_new.get_rect()
            cv.rectangle(self.__img, (x, y), (x + w, y + h), 0, -1)

        # text_blocks = [cc.get_rect() for cc in ccs]
        text_blocks = ccs.copy()

        if self.__debug:
            img_blocks = self.src.copy()
            for text_block in text_blocks:
                x, y, w, h = text_block.get_rect()
                cv.rectangle(img_blocks, (x, y), (x + w, y + h), (255, 0, 0), 4)
            iu.show_and_wait('Text Blocks', img_blocks)

        return text_blocks

    def __filter_ws(self):
        kernel = np.ones((1, 5), np.uint8)
        self.__img = cv.morphologyEx(self.__img, cv.MORPH_CLOSE, kernel, iterations=4)
        if self.__debug:
            iu.show_and_wait('Horizontal Closing', self.__img)

        ws_filter = WhiteSpaceFilter(self.__img, self.src, self.__debug)

        hr = MllClassifier(self.__img).get_region()
        self.__img = ws_filter.filter_ws(hr)

        self.__get_bounding_box_text_img(1)

    def __segment_paragraphs(self, text_blocks):
        new_hrs = []
        for text_block in text_blocks:
            next_hrs = self.__segment_region_paragraphs(text_block)
            new_hrs.extend(next_hrs)

        if self.__debug:
            img_blocks = self.src.copy()
            for hr in new_hrs:
                x, y, w, h = hr.get_rect()
                cv.rectangle(img_blocks, (x, y), (x + w, y + h), (255, 0, 255), 4)
            iu.show_and_wait('Paragraphs', img_blocks)

        return new_hrs

    def __segment_region_paragraphs(self, text_block):
        hr = self.__get_text_block_region(text_block)

        if len(hr.get_ccs()) <= 3:
            return [hr]

        page_width = self.__img.shape[1]
        block_width = max(hr.get_ccs(), key=lambda cc: cc.get_rect()[2]).get_rect()[2]
        if block_width < page_width / 8:
            return [hr]

        return self.__segment_paragraph(hr)

    @staticmethod
    def __segment_paragraph(hr):
        prev = 0
        curr = 1
        nextt = 2

        lines = [cc.get_rect() for cc in hr.get_ccs()]

        splits = []
        while nextt < len(lines):
            x_p, y_p, w_p, h_p = lines[prev]
            x_c, y_c, w_c, h_c = lines[curr]
            x_n, y_n, w_n, h_n = lines[nextt]
            th = h_p
            if w_c <= w_p - th:
                if w_n <= w_p - th:
                    splits.append(lines[curr][1] + lines[curr][3])
                    splits.append(lines[nextt][1])
                elif w_c <= w_n - th:
                    if x_c - th >= x_n:
                        splits.append(lines[prev][1] + lines[prev][3])
                        splits.append(lines[curr][1])
                    elif x_c + w_c < x_n + w_n - th:
                        splits.append(lines[curr][1] + lines[curr][3])
                        splits.append(lines[nextt][1])
                    pass
            prev += 1
            curr += 1
            nextt += 1

        split_hrs = hr.split_horizontally_at(splits)

        return split_hrs

    def __smooth_regions(self, hrs):
        ccs_text = []
        for hr in hrs:
            ccs = self.__smooth_region(hr)
            ccs_text.extend(ccs)

        blank = np.zeros(self.__img.shape[:2], np.uint8)
        self.__img = cv.drawContours(blank, [cc.get_contour() for cc in ccs_text], -1, 255, -1)

        if self.__debug:
            iu.show_and_wait('Bounding Box (Smoothed)', self.__img)

        return ccs_text

    @staticmethod
    def __smooth_region(hr: Region):
        img = hr.get_img()

        kernel_height = 3
        wl = [wl_i[3] for wl_i in hr.get_wl_h()]
        if len(wl) > 0 and np.max(wl) > 0:
            kernel_height = round(2 * np.percentile(wl, 75))

        kernel_width = 1
        # ws = hr.set_features().get_features()['ws']
        # if len(ws) > 0 and np.max(ws) > 0:
        #     kernel_width = round(2 * np.percentile(ws, 75))

        kernel = np.ones((kernel_height, kernel_width), np.uint8)
        # kernel = np.ones((5, 1), np.uint8)
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=4)

        return get_connected_components(closing, (hr.get_rect()[0], hr.get_rect()[1]))

    def __get_bounding_box_text_img(self, c=0):
        text_lines = self.__extract_text_lines()

        blank = np.zeros(self.__img.shape[0:2], np.uint8)
        for text_line in text_lines:
            x, y, w, h = text_line
            cv.rectangle(blank, (x, y + c), (x + w, y + h - c), 255, -1)

        self.__img = blank

        if self.__debug:
            iu.show_and_wait('Bounding Box', self.__img)

        return self.__img

    def __extract_text_lines(self):
        h, w = self.__img.shape[:2]
        region = Region((0, 0, w, h), self.__img)
        hcs = region.get_hcs()
        text_lines = []

        for hc in hcs:
            hc_rects = [cc.get_rect() for cc in hc]
            i = 0
            while i < len(hc_rects):
                x, y, w, h = hc_rects[i]
                j = i + 1
                while j < len(hc_rects):
                    x2, y2, w2, h2 = hc_rects[j]
                    if np.max([y, y2]) - np.min([y + h, y2 + h2]) < 0 and np.abs(x - (x2 + w2)) <= THETA * np.max(
                            [h, h2]) and np.max([h, h2]) <= 2 * np.min([h, h2]):
                        rects_union = union(hc_rects[i], hc_rects[j])
                        hc_rects.pop(j)
                        hc_rects.pop(i)
                        hc_rects.insert(i, rects_union)
                    else:
                        break
                    j += 1
                i += 1
            text_lines.extend(hc_rects)

        return text_lines

    def __get_text_block_region(self, text_block):
        blank = np.zeros(self.__img.shape, np.uint8)
        mask = cv.drawContours(blank, [text_block.get_contour()], -1, 255, -1)
        img_hr = self.__img * mask
        hr = Region(text_block.get_rect(), img_hr)

        hcs = hr.get_hcs()

        has_one_sentence_in_each_line = True
        for hc in hcs:
            if len(hc) > 1:
                has_one_sentence_in_each_line = False
                rect = hc[0].get_rect()
                for cc in hc[1:]:
                    rect = union(rect, cc.get_rect())
                cv.rectangle(self.__img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, -1)
                cv.rectangle(img_hr, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), 255, -1)

        if not has_one_sentence_in_each_line:
            hr = Region(text_block.get_rect(), img_hr)

        return hr
