import cv2 as cv
import numpy as np
from mll_classifier.mll_classifier import MllClassifier


THETA = 1.2


class TextSegmenter:
    def __init__(self, img, ccs_text, ccs_non_text):
        super().__init__()
        self.__img = img.copy()
        self.__ccs_text = ccs_text
        self.__ccs_non_text = ccs_non_text

    def segment_text(self):
        text_lines = self.__extract_text_lines()
        self.__bounding_box_text_img = self.__get_bounding_box_text_img(
            text_lines)
        # cv.namedWindow('Contours', cv.WINDOW_FREERATIO)
        # cv.imshow('Contours', self.__bounding_box_text_img)
        # if cv.waitKey(0) & 0xff == 27:
        #     cv.destroyAllWindows()
        hrs = MllClassifier(
            self.__bounding_box_text_img).get_first_level_homogeneous_regions()

        # self.__segment_paragraphs(hrs)

        ccs_text = self.__smooth_regions(hrs)

        return ccs_text

    def __get_bounding_box_text_img(self, text_lines):
        img = np.zeros(self.__img.shape[0:2], np.uint8)

        for text_line in text_lines:
            x, y, w, h = text_line
            img = cv.rectangle(img, (x, y), (x + w, y + h), 255, -1)

        return img

    def __segment_paragraphs(self):
        hrs = self.__filter_regions_ws_rects()

        rects = []
        for hr in hrs:
            rects.extend(self.__segment_region_paragraphs(hr))

    def __extract_horizontal_chains(self, hr):
        i = 0
        hcs = []

        xs = hr['xs'].copy()
        ys = hr['ys'].copy()
        widths = hr['widths'].copy()
        heights = hr['heights'].copy()
        rnn = hr['rnn'].copy()
        lnn = hr['lnn'].copy()
        rnws = []

        while i < len(xs):
            j = i + 1
            rnws_i = []
            while rnn[i] >= 0 and j < len(xs):
                if xs[j] == rnn[i]:
                    rects_union = self.__union(
                        (xs[i], ys[i], widths[i], heights[i]), (xs[j], ys[j], widths[j], heights[j]))
                    xs.pop(j)
                    xs.pop(i)
                    xs.insert(i, rects_union[0])

                    ys.pop(j)
                    ys.pop(i)
                    ys.insert(i, rects_union[1])

                    widths.pop(j)
                    widths.pop(i)
                    widths.insert(i, rects_union[2])

                    heights.pop(j)
                    heights.pop(i)
                    heights.insert(i, rects_union[3])

                    rnn.pop(i)
                    rnn.insert(i, rnn[j])
                    rnn.pop(j)

                    lnn.pop(j)
                    lnn.insert(j, lnn[i])
                    lnn.pop(i)

                    rnws_i.append(i)
                else:
                    j += 1
            rnws.append(rnws_i)
            i += 1

        return [(xs[i], ys[i], widths[i], heights[i], rnws[i]) for i in range(len(xs))]

    def __segment_region_paragraphs(self, hr):

        rects = []

    def __filter_regions_ws_rects(self):
        hrs = MllClassifier(
            self.__bounding_box_text_img).get_first_level_homogeneous_regions()
        for hr in hrs:
            hcs = self.__extract_horizontal_chains(hrs)
            self.__filter_ws_rects(hcs, hr)
        return MllClassifier(self.__bounding_box_text_img).get_first_level_homogeneous_regions()

    def __filter_ws_rects(self, hcs, hr):
        hcs.sort(key=lambda el: el[1])
        for i, hc in enumerate(hcs):
            x, y, w, h, rnws_idxs = hc
            rnws = [hr['rnws'][rnws_idx] for rnws_idx in rnws_idxs]
            rnws_with_idxs = [(hr['rnws'][rnws_idx], rnws_idx)
                              for rnws_idx in rnws_idxs]

            n_widest_rects = rnws_with_idxs
            removed_rnws = []
            n = max(1, 8 * w / self.__bounding_box_text_img.shape[1])
            if n < len(rnws):
                rnws_with_idxs.sort(key=lambda el: el[0]).reverse()
                n_widest_rects = rnws_with_idxs[:n]
                removed_rnws = rnws_with_idxs[n:-1]

            surrounding = []
            if i > 0:
                surrounding.append(hcs[i - 1])
            if i < len(hcs) - 1:
                surrounding.append(hcs[i + 1])

            while j < len(n_widest_rects):
                is_isolated = True
                for k, cc in enumerate(hr['ccs']):
                    x2, y2, w2, h2 = hr['xs'][k], hr['ys'][k], hr['widths'][k], hr['heights'][k]
                    # if self.__is_horizontally_aligned((x, y, w, h), (x2, y2, w2, h2)) \
                    #         and self.__is_vertically_aligned((x, y, w, h), (x2, y2, w2, h2)):
                    is_intersected = False
                    for hc1 in surrounding:
                        x1, y1, w1, h1, rnws_idxs1 = hc1
                        is_intersected = is_intersected or len(
                            self.__intersection((x1, y1, w1, h1), (x2, y2, w2, h2))) > 0

                    idx = n_widest_rects[j][1]
                    is_vertically_aligned_with_ws = hr['rnn'][idx] >= 0 and self.__is_vertically_aligned(
                        (hr['xs'][idx] + hr['widths']
                         [idx], 0, hr['rnn'][idx], 0),
                        (hr['xs'][k] + hr['widths'][k], 0, hr['rnn'][k], 0)
                    )
                    is_isolated = not (
                        is_intersected and is_vertically_aligned_with_ws)

                    if not is_isolated:
                        break

                if is_isolated:
                    removed_rnws.append(n_widest_rects.pop(j))
                else:
                    j += 1

            for removed_rnws_i in removed_rnws:
                cv.rectangle(
                    self.__bounding_box_text_img,
                    (hr['xs'][removed_rnws_i[1]], hr['ys'][removed_rnws_i[1]]),
                    (hr['xs'][removed_rnws_i[1]] + removed_rnws_i, hr['ys']
                     [removed_rnws_i[1]] + hr['heights'][removed_rnws_i[1]]),
                    (255, 255, 255),
                    -1
                )

    def __smooth_regions(self, hrs):
        ccs_text = []
        for hr in hrs:
            ccs = self.__smooth_region(hr)
            ccs_text.extend(ccs)
        return ccs_text

    def __smooth_region(self, hr):
        x1, y1, x2, y2 = hr['region']
        img = self.__crop_img(x1, y1, x2, y2)

        kernel_height = 2
        if len(hr['w']) > 0:
            kernel_height = int(
                2 * np.percentile(hr['w'], 75, interpolation='midpoint'))

        kernel_width = 2
        if len(hr['ws']) > 0:
            kernel_width = int(
                2 * np.percentile(hr['ws'], 75, interpolation='midpoint'))

        kernel = np.ones((kernel_height, kernel_width), np.uint8)

        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        ccs, _ = cv.findContours(
            closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(x1, y1))[-2:]
        return ccs

    def __extract_text_lines(self):
        i = 0
        rects = []
        for cc in self.__ccs_text:
            rects.append(cv.boundingRect(cc))

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
