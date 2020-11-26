import cv2 as cv
import numpy as np
from connected_components.connected_components import (
    ConnectedComponent, get_connected_components)
from mll_classifier.mll_classifier import MllClassifier
from mll_classifier.region import Region

THETA = 1.2


class TextSegmenter:
    def __init__(self, img, ccs_text, ccs_non_text):
        super().__init__()
        self.__img = img.copy()
        self.__ccs_text = ccs_text
        self.__ccs_non_text = ccs_non_text

    def segment_text(self):
        self.__bounding_box_text_img = self.__get_bounding_box_text_img()

        cv.namedWindow('Bounding Box', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box', self.__bounding_box_text_img)

        self.__segment_paragraphs()

        cv.namedWindow('Bounding Box WS Filtered', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box WS Filtered', self.__bounding_box_text_img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

        # ccs_text = self.__smooth_regions(regions)

        return ccs_text

    def __get_bounding_box_text_img(self):
        text_lines = self.__extract_text_lines()

        img = np.zeros(self.__img.shape[0:2], np.uint8)

        for text_line in text_lines:
            x, y, w, h = text_line
            img = cv.rectangle(img, (x, y), (x + w, y + h), 255, -1)

        return img

    def __segment_paragraphs(self):
        hrs = self.__filter_regions_ws_rects()

        # rects = []
        # for hr in hrs:
        #     rects.extend(self.__segment_region_paragraphs(hr))

    def __segment_region_paragraphs(self, hr):

        rects = []

    def __filter_regions_ws_rects(self):
        h, w = self.__bounding_box_text_img.shape[:2]
        region = Region((0, 0, w, h), self.__bounding_box_text_img)
        _, hrs = region.get_next_level_homogeneous_regions()
        for hr in hrs:
            hcs = self.__extract_horizontal_chains(hr)
            self.__filter_ws_rects(hcs, hr)
        return Region((0, 0, w, h), self.__bounding_box_text_img).get_next_level_homogeneous_regions()[1]

    def __extract_horizontal_chains(self, hr):
        hr.set_features()
        ccs = hr.get_ccs().copy()

        hcs = []
        hc = []

        while len(ccs) != 0:
            cc = ccs[0]
            while cc.get_lnn() != None and cc.get_lnn() in ccs:
                cc = cc.get_lnn()

            while cc != None and cc in ccs:
                hc.append(cc)
                ccs.remove(cc)
                cc = cc.get_rnn()
            hcs.append(hc)
            hc = []

        hcs.sort(key=lambda hc: min(
            hc, key=lambda cc: cc.get_rect()[1]).get_rect()[1])

        return hcs

    def __filter_ws_rects(self, hcs, hr):
        # hcs.sort(key=lambda hc: min(hc, key=lambda cc: cc.get_rect()[1]))
        hcs_rects = [[cc.get_rect() for cc in hc] for hc in hcs]
        for i, hc_rects in enumerate(hcs_rects):
            hc_rects: list
            hc: list = hcs[i]

            chain_width = (hc[-1].get_rect()[0] +
                           hc[-1].get_rect()[2]) - hc[0].get_rect()[0]
            page_width = self.__bounding_box_text_img.shape[1]

            n = np.max([1, int(8 * chain_width / page_width)])

            num_ws = len(hc) - 1
            while num_ws >= n:
                cc_rnws_min = min(hc, key=lambda cc: cc.get_rnws())
                idx = hc.index(cc_rnws_min)
                rects_union = self.__union(
                    hc_rects[idx], self.__get_ws_rect(hc[idx]))
                # hc.pop(idx)
                # hc_rects.pop(idx)
                # hc_rects.pop(idx)
                # hc_rects.insert(idx, rects_union)
                num_ws -= 1
                self.__remove_ws(rects_union)

            if i == len(hcs) - 1:
                break

            ws_curr_idx = 1
            while ws_curr_idx < len(hc) - 1:
                ws = self.__get_ws_rect(hc[ws_curr_idx])
                surroundings = []

                surroundings.append(hcs[i - 1])
                surroundings.append(hcs[i + 1])

                is_isolated = True

                for surrounding in surroundings:
                    for k in range(len(surrounding) - 1):
                        ws2 = self.__get_ws_rect(surrounding[k])
                        if self.__is_vertically_aligned(ws, ws2):
                            is_isolated = False
                            break
                    if not is_isolated:
                        ws_curr_idx += 1
                        break
                if is_isolated:
                    rects_union = self.__union(
                        hc_rects[ws_curr_idx], self.__get_ws_rect(hc[ws_curr_idx]))
                    hc.pop(ws_curr_idx)
                    hc_rects.pop(ws_curr_idx)
                    hc_rects.pop(ws_curr_idx)
                    hc_rects.insert(ws_curr_idx, rects_union)
                    num_ws -= 1
                    self.__remove_ws(rects_union)

    def __remove_ws(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.__bounding_box_text_img,
                     (x, y), (x + w, y + h), 255, -1)

    def __get_ws_rect(self, cc: ConnectedComponent):
        x, y, w, h = cc.get_rect()
        rnws = cc.get_rnws()
        return (x + w, y, rnws, h)

    def __smooth_regions(self, hrs):
        ccs_text = []
        for hr in hrs:
            ccs = self.__smooth_region(hr)
            ccs_text.extend(ccs)
        return ccs_text

    def __smooth_region(self, hr: Region):
        x, y, w, h = hr.get_rect()
        img = self.__crop_img(x, y, x + w, y + h)

        kernel_height = 2
        wl = [wl_i[3] for wl_i in hr.get_wl_h()]
        if len(hr) > 0:
            kernel_height = int(
                2 * np.percentile(wl, 75, interpolation='midpoint'))

        kernel_width = 2
        ws = hr.get_features()['ws']
        if len(ws) > 0:
            kernel_width = int(
                2 * np.percentile(ws, 75, interpolation='midpoint'))

        kernel = np.ones((kernel_height, kernel_width), np.uint8)

        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

        return get_connected_components(closing, (x, y))

    def __extract_text_lines(self):
        i = 0
        rects = [cc.get_rect() for cc in self.__ccs_text]

        while i < len(rects):
            x, y, w, h = rects[i]
            j = i + 1
            while j < len(rects):
                x2, y2, w2, h2 = rects[j]
                if np.max([y, y2]) - np.min([y + h, y2 + h2]) < 0 \
                        and np.abs(x - (x2 + w2)) <= THETA * np.max([h, h2]) \
                        and np.max([h, h2]) <= 2 * np.min([h, h2]):
                    rects_union = self.__union(rects[i], rects[j])
                    rects.pop(j)
                    rects.pop(i)
                    rects.insert(i, rects_union)
                else:
                    j += 1
            i += 1

        return rects

    def __union(self, a, b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0] + a[2], b[0] + b[2]) - x
        h = max(a[1] + a[3], b[1] + b[3]) - y
        return (x, y, w, h)

    def __intersection(self, a, b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        h = min(a[1] + a[3], b[1] + b[3]) - y
        if w < 0 or h < 0:
            return ()  # or (0,0,0,0) ?
        return (x, y, w, h)
        # Please remember a and b are rects.

    def __is_horizontally_aligned(self, a, b):
        y = max(a[1], b[1])
        h = min(a[1] + a[3], b[1] + b[3]) - y
        return h >= 0

    def __is_vertically_aligned(self, a, b):
        x = max(a[0], b[0])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        return w >= 0

    def __crop_img(self, x1, y1, x2, y2):
        return self.__bounding_box_text_img[y1:y2, x1:x2]
