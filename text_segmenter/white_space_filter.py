import cv2 as cv
import numpy as np

from connected_components.connected_components import ConnectedComponent, is_vertically_aligned
from mll_classifier.mll_classifier import Region


class WhiteSpaceFilter:

    def __init__(self, img, src):
        self.__img = img.copy()
        self.src = src.copy()

    def filter_ws(self, hr: Region):
        # self.__ccs_text_large = self.__remove_large_text_ccs(hrs)
        self.__filter_ws_rects(hr)
        # TODO: Remove
        # cv.namedWindow('Color filtered', cv.WINDOW_FREERATIO)
        # cv.imshow('Color filtered', self.src)
        # if cv.waitKey(0) & 0xff == 27:
        #     cv.destroyAllWindows()

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

    def __filter_ws_rects(self, hr: Region):
        hcs = hr.get_hcs()

        # TODO: Remove
        for i, hc in enumerate(hcs):
            color = [0, 0, 0]
            color[i % 3] = 255
            for cc in hc:
                x, y, w, h = cc.get_rect()
                cv.rectangle(self.src, (x, y), (x + w, y + h), tuple(color), -1)

        only_single_cc_chains = True
        for hc in hcs:
            if len(hc) > 1:
                only_single_cc_chains = False
                break
        if only_single_cc_chains:
            return

        ws_rects = WhiteSpaceFilter.__get_hcs_ws_rects(hcs)

        self.__filter_small_and_isolated_ws(hcs, ws_rects)
        for i in range(2):
            self.__filter_within_col_candidates(ws_rects, i)

    def __filter_within_col_candidates(self, ws_rects, run):
        ws_max_gaps_pairs = self.__group_ws(ws_rects, run)
        for col in ws_max_gaps_pairs:
            is_within_col = WhiteSpaceFilter.__is_within_col_candidates_chain(col)

            if is_within_col:
                for ws_max_gap_pair in col:
                    # cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, ws_max_gap_pair['rect'])
                    # ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
                    # ws_sum = np.sum(cropped_ws)
                    # if ws_length > 0 and ws_sum / ws_length != 255:
                    self.__remove_ws(ws_max_gap_pair['rect'])
                    self.__remove_ws_color(ws_max_gap_pair['rect'], (0, 0, 100))
                    ws_max_gap_pair['is_removed'] = True
            elif run > 0:
                self.__filter_top_and_bottom_within_col_candidates(col)

    def __filter_top_and_bottom_within_col_candidates(self, col):
        if col[0]['is_within_col_candidate']:
            # cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, col[0]['rect'])
            # ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
            # ws_sum = np.sum(cropped_ws)
            # if ws_length > 0 and ws_sum / ws_length != 255:
            self.__remove_ws(col[0]['rect'])
            self.__remove_ws_color(col[0]['rect'], (100, 0, 100))
        if col[-1]['is_within_col_candidate']:
            # cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, col[-1]['rect'])
            # ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
            # ws_sum = np.sum(cropped_ws)
            # if ws_length > 0 and ws_sum / ws_length != 255:
            self.__remove_ws(col[-1]['rect'])
            self.__remove_ws_color(col[-1]['rect'], (100, 0, 100))

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
                if is_vertically_aligned(ws['rect'], ws2['rect']):
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

                gaps = []
                j = i - 1
                while j >= 0:
                    if not row[j]['is_removed']:
                        gaps.append(row[j]['rect'][2])
                        break
                    j -= 1

                j = i + 1
                while j < len(row):
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
        for i, row in enumerate(ws_rects):
            self.__remove_small_ws(hcs[i], ws_rects, i)
            self.__remove_isolated_ws(hcs, ws_rects, i)

    def __remove_small_ws(self, hc, ws_rects, i):
        chain_width = (hc[-1].get_rect()[0] +
                       hc[-1].get_rect()[2]) - hc[0].get_rect()[0]
        page_width = self.__img.shape[1]

        n = np.max([1, round(8 * chain_width / page_width)])

        ws_rects_row = ws_rects[i]
        num_ws = len(ws_rects_row)
        while num_ws > n:
            rnws_min = min(ws_rects_row, key=lambda ws: ws['rect'][2])
            # cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, rnws_min['rect'])
            # ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
            # ws_sum = np.sum(cropped_ws)
            # if ws_length > 0 and ws_sum / ws_length != 255:
            self.__remove_ws(rnws_min['rect'])
            self.__remove_ws_color(rnws_min['rect'], (100, 0, 0))
            rnws_min['is_removed'] = True
            rnws_min['is_small'] = True
            num_ws -= 1

    def __remove_isolated_ws(self, hcs, ws_rects, i):
        # if i == 0 or i == len(ws_rects) - 1:
        #     return
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
                if not ws['is_removed'] and is_vertically_aligned(ws_rects_row[ws_curr_idx]['rect'], ws['rect']):
                    is_isolated = False
                    break

            if is_isolated:
                for idx in surr_idx:
                    if not (hcs[idx][0].get_rect()[0] <= ws_rects_row[ws_curr_idx]['rect'][0] and
                            hcs[idx][-1].get_rect()[0] + hcs[idx][-1].get_rect()[2] >=
                            ws_rects_row[ws_curr_idx]['rect'][0] + ws_rects_row[ws_curr_idx]['rect'][2]):
                        is_isolated = False
                        break

            if is_isolated:
                # cropped_ws = WhiteSpaceFilter.__crop_img(self.__img, ws_rects_row[ws_curr_idx]['rect'])
                # ws_length = cropped_ws.shape[0] * cropped_ws.shape[1]
                # ws_sum = np.sum(cropped_ws)
                # if ws_length > 0 and ws_sum / ws_length != 255:
                self.__remove_ws(ws_rects_row[ws_curr_idx]['rect'])
                self.__remove_ws_color(ws_rects_row[ws_curr_idx]['rect'], (0, 100, 0))
                ws_rects_row[ws_curr_idx]['is_removed'] = True

            ws_curr_idx += 1

    def __remove_ws(self, rect):
        x, y, w, h = rect
        cv.rectangle(self.__img, (x, y), (x + w, y + h), 255, -1)

    def __remove_cc(self, cc):
        x, y, w, h = cc.get_rect()
        cv.rectangle(self.__img, (x, y), (x + w, y + h), 0, -1)

    # TODO: REMOVE
    def __remove_ws_color(self, rect, color):
        x, y, w, h = rect
        cv.rectangle(self.src, (x, y), (x + w, y + h), color, -1)

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
