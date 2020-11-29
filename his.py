import cv2 as cv
import numpy as np
from connected_components.connected_components import (
    ConnectedComponent, get_connected_components)
from mll_classifier.region import Region
from mll_classifier.mll_classifier import MllClassifier


THETA = 1


'''
    This class contains many methods that depends on mutable arrays
    and it needs to be refactored quite a bit.
'''


class TextSegmenter:
    def __init__(self, img, ccs_text, src):
        super().__init__()
        self.__img = img.copy()
        self.__ccs_text = ccs_text
        self.src = src.copy()

    def segment_text(self):
        self.__bounding_box_text_img = self.__get_bounding_box_text_img()

        cv.namedWindow('Bounding Box', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box', self.__bounding_box_text_img)

        ccs_text = self.__segment_paragraphs()

        cv.namedWindow('Bounding Box Smoothed', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box Smoothed', self.__bounding_box_text_img)

        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

        return ccs_text

    def __get_bounding_box_text_img(self):
        text_lines = self.__extract_text_lines()

        img = np.zeros(self.__img.shape[0:2], np.uint8)

        for text_line in text_lines:
            x, y, w, h = text_line
            img = cv.rectangle(img, (x, y), (x + w, y + h), 255, -1)

        return img

    def __segment_paragraphs(self):
        # self.__ccs_text_large = self.__remove_large_text_ccs(hrs)

        num_removed = 1
        while num_removed > 0:
            hrs = MllClassifier(
                self.__bounding_box_text_img).get_next_level_homogeneous_regions()
            num_removed = self.__filter_regions_ws_rects(hrs)

        cv.namedWindow('Bounding Box WS Filtered', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box WS Filtered', self.__bounding_box_text_img)

        cv.namedWindow('Color filtered', cv.WINDOW_FREERATIO)
        cv.imshow('Color filtered', self.src)

        ccs = get_connected_components(self.__bounding_box_text_img)

        for cc in ccs:
            x, y, w, h = cc.get_rect()
            cv.rectangle(self.__bounding_box_text_img,
                         (x, y), (x + w, y + h), 255, -1)

        cv.namedWindow('Bounding Box WS Filtered 2', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box WS Filtered 2', self.__bounding_box_text_img)

        hrs = MllClassifier(
            self.__bounding_box_text_img).get_next_level_homogeneous_regions()

        return self.__smooth_regions(hrs)

    def __remove_large_text_ccs(self, hrs):
        ccs_text_large = []
        for hr in hrs:
            _, _, w, h = hr.get_rect()
            page_long_side = np.max([h, w])
            for cc in hr.get_ccs():
                rect = cc.get_rect()
                cc_long_side = np.max([rect[2], rect[3]])
                if cc_long_side > 0.1 * page_long_side:
                    ccs_text_large.append(cc)
                    self.__remove_cc(cc)
                    hr.get_ccs().remove(cc)
        print(f'Number of large ccs: {len(ccs_text_large)}')
        return ccs_text_large

    def __filter_regions_ws_rects(self, hrs):
        num_removed = 0
        for hr in hrs:
            num_removed += self.__filter_ws_rects(hr)

        return num_removed

    def __filter_ws_rects(self, hr):
        hcs = self.__extract_horizontal_chains(hr)
        ws_rects = self.__get_hcs_ws_rects(hcs)

        num_removed = self.__filter_small_and_isolated_ws(hcs, ws_rects)
        num_removed += self.__filter_within_col_candidates(ws_rects)

        return num_removed

    def __filter_within_col_candidates(self, ws_rects):
        ws_max_gaps_pairs = self.__group_ws(ws_rects)
        num_removed = 0
        for col in ws_max_gaps_pairs:
            is_within_col = self.__filter_within_col_candidates_chain(col)

            if is_within_col:
                for ws_max_gap_pair in col:
                    self.__remove_ws(ws_max_gap_pair['rect'])
                    self.__remove_ws_col(ws_max_gap_pair['rect'])
                num_removed += len(col)
                ws_max_gaps_pairs.remove(col)
            else:
                num_removed += self.__filter_top_and_bottom_within_col_candidates(
                    col)

        return num_removed

    def __filter_within_col_candidates_chain(self, col):
        is_within_col = True
        for ws_max_gap_pair in col:
            if not ws_max_gap_pair['is_within_col_candidate']:
                is_within_col = False
                break
        return is_within_col

    def __filter_top_and_bottom_within_col_candidates(self, col):
        num_removed = 0
        if col[0]['is_within_col_candidate']:
            self.__remove_ws(col[0]['rect'])
            self.__remove_ws_top(col[0]['rect'])
            num_removed += 1
        if col[-1]['is_within_col_candidate']:
            self.__remove_ws(col[-1]['rect'])
            self.__remove_ws_top(col[-1]['rect'])
            num_removed += 1

        return num_removed

    def __group_ws(self, ws_rects):
        i = 0
        ws_vcs = []
        ws_vc = []
        ws_rects = [item for row in ws_rects for item in row]
        in_chain = [False for item in ws_rects]
        while i < len(ws_rects):
            if in_chain[i]:
                continue

            ws_vc.append(ws_rects[i])
            in_chain[i] = True
            j = 0
            while j < len(ws_rects):
                if self.__is_vertically_aligned(ws_rects[i]['rect'], ws_rects[j]['rect']):
                    ws_vc.append(ws_rects[j])
                    in_chain[j] = True
                j += 1

            ws_vcs.append(ws_vc)
            ws_vc = []
            i += 1

        return self.__convert_to_ws_max_gap_pairs(ws_vcs)

    def __convert_to_ws_max_gap_pairs(self, ws_rects):
        for row in ws_rects:
            for i, ws in enumerate(row):
                if ws['is_removed']:
                    continue
                if i > 0 and i < len(row) - 1 \
                        and row[i - 1]['is_small'] and row[i + 1]['is_small']:
                    ws['is_within_col_candidate'] = True
                    continue

                j = i - 1
                gaps = []
                while j > 0:
                    if not row[j]['is_removed']:
                        gaps.append(row[j]['rect'][2])
                        break
                    j += 1

                j = i + 1
                while j < len(row) - 1:
                    if not row[j]['is_removed']:
                        gaps.append(row[j]['rect'][2])
                        break
                    j += 1

                max_gap = 0
                if len(gaps) > 0:
                    max_gap = max(gaps)

                ws['is_within_col_candidate'] = ws['rect'][2] <= max_gap

        return ws_rects

    def __filter_small_and_isolated_ws(self, hcs, ws_rects):
        num_removed = 0
        for i, row in enumerate(ws_rects):
            num_removed += self.__remove_small_ws(hcs[i], row)

            if i == len(ws_rects) - 1:
                break

            num_removed += self.__remove_isolated_ws(ws_rects, row, i)

        return num_removed

    def __remove_small_ws(self, hc, ws_rects_row):
        num_removed = 0
        chain_width = (hc[-1].get_rect()[0] +
                       hc[-1].get_rect()[2]) - hc[0].get_rect()[0]
        page_width = self.__bounding_box_text_img.shape[1]

        n = np.max([1, round(8 * chain_width / page_width)])

        num_ws = len(ws_rects_row)
        while num_ws > n:
            rnws_min = min(ws_rects_row, key=lambda ws: ws['rect'][2])
            self.__remove_ws(rnws_min['rect'])
            self.__remove_ws_small(rnws_min['rect'])
            rnws_min['is_removed'] = True
            rnws_min['is_small'] = True
            num_ws -= 1
            num_removed += 1

        return num_removed

    def __remove_isolated_ws(self, ws_rects, ws_rects_row, i):
        num_removed = 0
        ws_curr_idx = 0
        while ws_curr_idx < len(ws_rects_row):
            if ws_rects_row[ws_curr_idx]['is_removed']:
                ws_curr_idx += 1
                continue
            surroundings = []

            surroundings.append(ws_rects[i - 1])
            surroundings.append(ws_rects[i + 1])

            is_isolated = True

            for surrounding in surroundings:
                for k in range(len(surrounding)):
                    if self.__is_vertically_aligned(ws_rects_row[ws_curr_idx]['rect'], surrounding[k]['rect']):
                        is_isolated = False
                        break
                if not is_isolated:
                    ws_curr_idx += 1
                    break
            if is_isolated:
                self.__remove_ws(ws_rects_row[ws_curr_idx]['rect'])
                self.__remove_ws_iso(ws_rects_row[ws_curr_idx]['rect'])
                ws_rects_row[ws_curr_idx]['is_removed'] = True
                num_removed += 1

        return num_removed

    def __remove_ws(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.__bounding_box_text_img,
                     (x, y), (x + w, y + h), 255, -1)

    def __remove_ws_iso(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.src,
                     (x, y), (x + w, y + h), (255, 0, 0), -1)

    def __remove_ws_small(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.src,
                     (x, y), (x + w, y + h), (0, 255, 0), -1)

    def __remove_ws_col(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.src,
                     (x, y), (x + w, y + h), (0, 0, 0), -1)

    def __remove_ws_top(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.src,
                     (x, y), (x + w, y + h), (255, 0, 255), -1)

    def __remove_cc(self, cc):
        x, y, w, h = cc.get_rect()
        cv.rectangle(self.__bounding_box_text_img,
                     (x, y), (x + w, y + h), 0, -1)

    def __get_hcs_ws_rects(self, hcs):
        return [[{
            'rect': self.__get_ws_rect(cc),
            'is_removed': False,
            'is_small': False,
            'is_within_col_candidate': False
        } for cc in hc[:-1]] for hc in hcs]

    def __get_ws_rect(self, cc: ConnectedComponent):
        x, y, w, h = cc.get_rect()
        rnws = cc.get_rnws()
        return (x + w, y, rnws, h)

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

    def __smooth_regions(self, hrs):
        ccs_text = []
        for hr in hrs:
            ccs = self.__smooth_region(hr)
            ccs_text.extend(ccs)

        blank = np.zeros((self.__bounding_box_text_img.shape[:2]))
        self.__bounding_box_text_img = cv.drawContours(
            blank, [cc.get_contour() for cc in ccs_text], -1, 255, -1)

        return ccs_text

    def __smooth_region(self, hr: Region):
        x, y, w, h = hr.get_rect()
        img = self.__crop_img(x, y, x + w, y + h)

        kernel_height = 2
        wl = [wl_i[3] for wl_i in hr.get_wl_h()]
        if len(wl) > 0:
            kernel_height = round(
                2 * np.percentile(wl, 75, interpolation='midpoint'))

        kernel_width = 2
        ws = hr.set_features().get_features()['ws']
        if len(ws) > 0:
            kernel_width = round(
                2 * np.percentile(ws, 75, interpolation='midpoint'))

        kernel = np.ones((kernel_height, kernel_width), np.uint8)

        # kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_height, kernel_width))

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

    # def __intersection(self, a, b):
    #     x = max(a[0], b[0])
    #     y = max(a[1], b[1])
    #     w = min(a[0] + a[2], b[0] + b[2]) - x
    #     h = min(a[1] + a[3], b[1] + b[3]) - y
    #     if w < 0 or h < 0:
    #         return ()  # or (0,0,0,0) ?
    #     return (x, y, w, h)
    #     # Please remember a and b are rects.

    # def __is_horizontally_aligned(self, a, b):
    #     y = max(a[1], b[1])
    #     h = min(a[1] + a[3], b[1] + b[3]) - y
    #     return h >= 0

    def __is_vertically_aligned(self, a, b):
        x = max(a[0], b[0])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        return w >= 0

    def __crop_img(self, x1, y1, x2, y2):
        return self.__bounding_box_text_img[y1:y2, x1:x2]
