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
        self.__bounding_box_text_img = self.__get_bounding_box_text_img(text_lines)
        # cv.namedWindow('Contours', cv.WINDOW_FREERATIO)
        # cv.imshow('Contours', self.__bounding_box_text_img)
        # if cv.waitKey(0) & 0xff == 27:
        #     cv.destroyAllWindows()
        hrs = MllClassifier(
            self.__bounding_box_text_img).get_first_level_homogeneous_regions()

        self.__segment_paragraphs()

        ccs_text = []
        for hr in hrs:
            ccs = self.__smooth_region(hr)
            ccs_text.extend(ccs)

        return ccs_text

    def __get_bounding_box_text_img(self, text_lines):
        img = np.zeros(self.__img.shape[0:2], np.uint8)

        for text_line in text_lines:
            x, y, w, h = text_line
            img = cv.rectangle(img, (x, y), (x + w, y + h), 255, -1)

        return img

    def __segment_paragraphs(self):
        pass

    def __smooth_region(self, hr):
        x1, y1, x2, y2 = hr['region']
        img = self.__crop_img(x1, y1, x2, y2)

        kernel_width = 2
        if len(hr['ws']) > 0:
            kernel_width = int(
                2 * np.percentile(hr['ws'], 75, interpolation='midpoint'))

        kernel_height = 2
        if len(hr['w']) > 0:
            kernel_height = int(
                2 * np.percentile(hr['w'], 75, interpolation='midpoint'))
        kernel = np.ones((kernel_width, kernel_height), np.uint8)

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

    def __crop_img(self, x1, y1, x2, y2):
        return self.__bounding_box_text_img[y1:y2, x1:x2]
