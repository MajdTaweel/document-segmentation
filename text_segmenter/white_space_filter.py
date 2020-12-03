from typing import List

import cv2 as cv
import numpy as np

from connected_components.connected_components import ConnectedComponent, union
from mll_classifier.mll_classifier import MllClassifier, Region

THETA = 1.2


class WhiteSpaceFilter:

    def __init__(self, img, src):
        self.__img = img.copy()
        self.src = src.copy()

    def filter_ws(self):
        # self.__ccs_text_large = self.__remove_large_text_ccs(hrs)
        self.__img = self.__get_bounding_box_text_img()
        cv.namedWindow('Bounding Box', cv.WINDOW_FREERATIO)
        cv.imshow('Bounding Box', self.__img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
        hrs = MllClassifier(self.__img).get_next_level_homogeneous_regions()
        self.__filter_regions_ws_rects(hrs)

        # for i, hr in enumerate(hrs):
        #     cv.namedWindow(f'HR {i}', cv.WINDOW_FREERATIO)
        #     cv.imshow(f'HR {i}', hr.get_img())
        cv.namedWindow('Color filtered', cv.WINDOW_FREERATIO)
        cv.imshow('Color filtered', self.src)

        return self.__img

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

    def __get_bounding_box_text_img(self):
        text_lines = self.__extract_text_lines()

        blank = np.zeros(self.__img.shape[0:2], np.uint8)
        for text_line in text_lines:
            x, y, w, h = text_line
            cv.rectangle(blank, (x, int(y + h / 4) - 1), (x + w, int(y + h / 2) + 1), 255, -1)

        self.__img = blank

        return self.__img

    def __filter_regions_ws_rects(self, hrs):
        num_removed = 0
        for hr in hrs:
            num_removed += self.__filter_ws_rects(hr)

        cv.namedWindow('Horizontal Chains', cv.WINDOW_FREERATIO)
        cv.imshow('Horizontal Chains', self.src)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

        return num_removed

    def __filter_ws_rects(self, hr: Region):
        hcs: List[List[ConnectedComponent]] = hr.get_hcs()

        for i, hc in enumerate(hcs):
            color = [0, 0, 0]
            color[i % 3] = 255
            for j, cc in enumerate(hc):
                if j == len(hc) - 1:
                    break
                x, y, w, h = cc.get_rect()
                cv.rectangle(self.src, (x, y), (x + w, y + h), tuple(color), -1)
                x, y, w, h = hc[j].get_rect()
                cv.rectangle(self.src, (x, y), (x + w, y + h), tuple(color), -1)

        only_single_cc_chains = True
        for hc in hcs:
            if len(hc) > 1:
                only_single_cc_chains = False
                break
        if only_single_cc_chains:
            return 0

        ws_rects = WhiteSpaceFilter.__get_hcs_ws_rects(hcs)

        num_removed = self.__filter_small_and_isolated_ws(hcs, ws_rects)
        for i in range(2):
            num_removed += self.__filter_within_col_candidates(ws_rects, i)

        return num_removed

    def __filter_within_col_candidates(self, ws_rects, run):
        ws_max_gaps_pairs = self.__group_ws(ws_rects, run)
        num_removed = 0
        for col in ws_max_gaps_pairs:
            is_within_col = WhiteSpaceFilter.__is_within_col_candidates_chain(col)

            if is_within_col:
                for ws_max_gap_pair in col:
                    cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, ws_max_gap_pair['rect'])
                    ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
                    ws_sum = np.sum(cropped_ws)
                    if ws_sum / ws_length != 255:
                        self.__remove_ws(ws_max_gap_pair['rect'])
                        self.__remove_ws_color(ws_max_gap_pair['rect'], (0, 0, 100))
                        num_removed += 1
                ws_max_gaps_pairs.remove(col)
            elif run > 0:
                num_removed += self.__filter_top_and_bottom_within_col_candidates(
                    col)

        return num_removed

    def __filter_top_and_bottom_within_col_candidates(self, col):
        num_removed = 0
        if col[0]['is_within_col_candidate']:
            cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, col[0]['rect'])
            ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
            ws_sum = np.sum(cropped_ws)
            if ws_sum / ws_length != 255:
                self.__remove_ws(col[0]['rect'])
                self.__remove_ws_color(col[0]['rect'], (100, 0, 100))
                num_removed += 1
        if col[-1]['is_within_col_candidate']:
            cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, col[-1]['rect'])
            ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
            ws_sum = np.sum(cropped_ws)
            if ws_sum / ws_length != 255:
                self.__remove_ws(col[-1]['rect'])
                self.__remove_ws_color(col[-1]['rect'], (100, 0, 100))
                num_removed += 1

        return num_removed

    def __group_ws(self, ws_rects, run):
        i = 0
        ws_vcs = []
        while i < len(ws_rects):
            j = 0
            while j < len(ws_rects[i]):
                ws = ws_rects[i][j]
                ws_vc = self.__get_ws_vc(ws_rects, i, ws)
                ws_vcs.append(ws_vc)
            i += 1

        return self.__convert_to_ws_max_gap_pairs(ws_vcs, run)

    @staticmethod
    def __get_ws_vc(ws_rects, i, ws):
        ws_vc = []
        ws_vc.append(ws)
        ws_rects[i].remove(ws)
        i += 1
        while i < len(ws_rects):
            j = 0
            while j < len(ws_rects[i]):
                ws2 = ws_rects[i][j]
                if WhiteSpaceFilter.__is_vertically_aligned(ws['rect'], ws2['rect']):
                    ws_vc.append(ws2)
                    ws_rects[i].pop(j)
                    break
                else:
                    j += 1
            i += 1
        return ws_vc

    @staticmethod
    def __convert_to_ws_max_gap_pairs(ws_rects, run):
        for row in ws_rects:
            for i, ws in enumerate(row):
                if ws['is_removed']:
                    continue
                if run > 0 and i > 0 and i < len(row) - 1 \
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
            num_removed += self.__remove_small_ws(hcs[i], ws_rects, i)
            num_removed += self.__remove_isolated_ws(hcs, ws_rects, i)

        return num_removed

    def __remove_small_ws(self, hc, ws_rects, i):
        num_removed = 0
        chain_width = (hc[-1].get_rect()[0] +
                       hc[-1].get_rect()[2]) - hc[0].get_rect()[0]
        page_width = self.__img.shape[1]

        n = np.max([1, round(8 * chain_width / page_width)])

        ws_rects_row = ws_rects[i]
        num_ws = len(ws_rects_row)
        while num_ws > n:
            rnws_min = min(ws_rects_row, key=lambda ws: ws['rect'][2])
            cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, rnws_min['rect'])
            ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
            ws_sum = np.sum(cropped_ws)
            if ws_sum / ws_length != 255:
                self.__remove_ws(rnws_min['rect'])
                self.__remove_ws_color(rnws_min['rect'], (100, 0, 0))
                num_removed += 1
            rnws_min['is_removed'] = True
            rnws_min['is_small'] = True
            num_ws -= 1

        return num_removed

    def __remove_isolated_ws(self, hcs, ws_rects, i):
        num_removed = 0
        ws_curr_idx = 0
        ws_rects_row = ws_rects[i]

        surrounding = []
        surr_idx = []
        if i > 0:
            surr_idx.append(i - 1)
            surrounding.extend(ws_rects[i - 1])
        if i < len(ws_rects) - 1:
            surrounding.extend(ws_rects[i + 1])
            surr_idx.append(i + 1)

        while ws_curr_idx < len(ws_rects_row):
            if ws_rects_row[ws_curr_idx]['is_removed']:
                ws_curr_idx += 1
                continue

            is_isolated = True

            for ws in surrounding:
                if not ws['is_removed'] and WhiteSpaceFilter.__is_vertically_aligned(ws_rects_row[ws_curr_idx]['rect'],
                                                                                     ws['rect']):
                    is_isolated = False
                    break

            if is_isolated:
                for idx in surr_idx:
                    if not (hcs[idx][0].get_rect()[0] <= ws_rects_row[ws_curr_idx]['rect'][0] and
                            hcs[idx][0].get_rect()[0] + hcs[idx][0].get_rect()[2] >=
                            ws_rects_row[ws_curr_idx]['rect'][0] + ws_rects_row[ws_curr_idx]['rect'][2]):
                        is_isolated = False
                        break

            if is_isolated:
                cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, ws_rects_row[ws_curr_idx]['rect'])
                ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
                ws_sum = np.sum(cropped_ws)
                if ws_sum / ws_length != 255:
                    self.__remove_ws(ws_rects_row[ws_curr_idx]['rect'])
                    self.__remove_ws_color(ws_rects_row[ws_curr_idx]['rect'], (0, 100, 0))
                    ws_rects_row[ws_curr_idx]['is_removed'] = True
                    num_removed += 1

            ws_curr_idx += 1

        return num_removed

    def __remove_ws(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.__img,
                     (x, y), (x + w, y + h), 255, -1)

    def __remove_cc(self, cc):
        x, y, w, h = cc.get_rect()
        cv.rectangle(self.__img,
                     (x, y), (x + w, y + h), 0, -1)

    def __remove_ws_color(self, rect, color):
        x, y, w, h = rect
        cv.rectangle(self.src, (x, y), (x + w, y + h), color, -1)

    def __extract_text_lines(self):
        i = 0
        h, w = self.__img.shape[:2]
        region = Region((0, 0, w, h), self.__img)
        hcs = region.get_hcs()
        text_lines = []

        ccs = region.get_ccs()

        cv.drawContours(self.src, [cc.get_contour() for cc in ccs], -1, (255, 0, 255), -1)
        for hc in hcs:
            cv.drawContours(self.src, [cc.get_contour() for cc in hc], -1, (255, 255, 0), -1)

        cv.namedWindow('Horizontal Chains', cv.WINDOW_FREERATIO)
        cv.imshow('Horizontal Chains', self.src)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

        for hc in hcs:
            hc_rects = [cc.get_rect() for cc in hc]
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
                        j += 1
                i += 1
            text_lines.extend(hc_rects)

        return text_lines

    @staticmethod
    def __is_vertically_aligned(a, b):
        x = max(a[0], b[0])
        w = min(a[0] + a[2], b[0] + b[2]) - x
        return w >= 0

    @staticmethod
    def __get_ws_rect(cc: ConnectedComponent):
        x, y, w, h = cc.get_rect()
        rnws = cc.get_rnws()
        return x + w, y, rnws, h

    @staticmethod
    def __get_hcs_ws_rects(hcs):
        return [[{
            'rect': WhiteSpaceFilter.__get_ws_rect(cc),
            'is_removed': False,
            'is_small': False,
            'is_within_col_candidate': False
        } for cc in hc[:-1]] for hc in hcs]

    @staticmethod
    def __crop_img(img, rect):
        x, y, w, h = rect
        return img[y:y + h, x:x + w]

    @staticmethod
    def __is_within_col_candidates_chain(col):
        for ws_max_gap_pair in col:
            if not ws_max_gap_pair['is_within_col_candidate']:
                return False
        return True
