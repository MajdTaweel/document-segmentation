from typing import List

import cv2 as cv
import numpy as np


class ConnectedComponent:
    def __init__(self, contour):
        super().__init__()
        self.__contour = contour
        self.__rect = cv.boundingRect(contour)
        self.__area = cv.contourArea(contour)
        self.__rect_area = self.__rect[2] * self.__rect[3]
        self.__dens = self.__area / self.__rect_area
        self.__lnn = None
        self.__rnn = None
        self.__lnws = 0
        self.__rnws = 0
        self.__num_rn = 0
        self.__num_ln = 0
        self.__hw_rate = np.min([self.__rect[2], self.__rect[3]]) / np.max([self.__rect[2], self.__rect[3]])

    def get_contour(self):
        return self.__contour

    def set_contour(self, contour):
        self.__init__(contour)

    def get_rect(self):
        return self.__rect

    def get_area(self):
        return self.__area

    def get_rect_area(self):
        return self.__rect_area

    def get_dens(self):
        return self.__dens

    def get_hw_rate(self):
        return self.__hw_rate

    def get_lnn(self):
        return self.__lnn

    def get_rnn(self):
        return self.__rnn

    def get_lnws(self):
        return self.__lnws

    def get_rnws(self):
        return self.__rnws

    def get_num_ln(self):
        return self.__num_ln

    def get_num_rn(self):
        return self.__num_rn

    def is_horizontally_aligned_with(self, cc: 'ConnectedComponent'):
        y = max(self.__rect[1], cc.get_rect()[1])
        h = min(self.__rect[1] + self.__rect[3],
                cc.get_rect()[1] + cc.get_rect()[3]) - y
        return h >= 0

    def contains(self, cc: 'ConnectedComponent'):
        x_i, y_i, w_i, h_i = self.get_rect()
        x_j, y_j, w_j, h_j = cc.get_rect()
        return x_j >= x_i and y_j >= y_i and x_j + w_j <= x_i + w_i and y_j + h_j <= y_i + h_i

    def set_num_ln(self, num_ln):
        self.__num_ln = num_ln

    def set_num_rn(self, num_rn):
        self.__num_rn = num_rn

    def set_lnn(self, lnn):
        self.__lnn = lnn
        self.__lnws = self.get_rect()[0] - (self.__lnn.get_rect()[0] + self.__lnn.get_rect()[2])

    def set_rnn(self, rnn):
        self.__rnn = rnn
        self.__rnws = self.__rnn.get_rect()[0] - (self.get_rect()[0] + self.get_rect()[2])


def get_connected_components(img, offset=None, external=False) -> List[ConnectedComponent]:
    method = cv.RETR_TREE
    if external:
        method = cv.RETR_EXTERNAL
    contours, _ = cv.findContours(
        img, method, cv.CHAIN_APPROX_SIMPLE, offset=offset)[-2:]
    return [ConnectedComponent(contour) for contour in contours]


def set_ccs_neighbors(ccs: List[ConnectedComponent]) -> List[List[ConnectedComponent]]:
    if len(ccs) == 0:
        return []

    if len(ccs) == 1:
        return [[ccs[0]]]

    ccs = ccs.copy()
    ccs.sort(key=lambda cc: cc.get_rect()[1])
    hcs = []
    cc1 = ccs[0]
    hc = [cc1]
    for i, cc2 in enumerate(ccs[1:]):
        if cc1.is_horizontally_aligned_with(cc2):
            hc.append(cc2)
            cc1 = cc2
        else:
            hc.sort(key=lambda cc: cc.get_rect()[0])
            hcs.append(hc)
            cc1 = cc2
            hc = [cc1]
        if i == len(ccs) - 2:
            hc.sort(key=lambda cc: cc.get_rect()[0])
            hcs.append(hc)

    for hc in hcs:
        for i, cc in enumerate(hc):
            cc.set_num_ln(i)
            cc.set_num_rn(len(hc) - 1 - i)

            if i > 0:
                cc.set_lnn(hc[i - 1])
            if i < len(hc) - 1:
                cc.set_rnn(hc[i + 1])

    return hcs


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return x, y, w, h


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()
    return x, y, w, h


def includes(img_shape, cc1, cc2):
    blank = np.zeros(img_shape)

    img1 = cv.drawContours(blank.copy(), [cc1.get_contour()], -1, 255, -1)
    img2 = cv.drawContours(blank.copy(), [cc2.get_contour()], -1, 255, -1)

    ccs_intersection = np.logical_and(img1, img2)

    return ccs_intersection.all()


def does_intersect(img_shape, cc1: ConnectedComponent, cc2: ConnectedComponent):
    blank = np.zeros(img_shape, np.uint8)

    img1 = cv.drawContours(blank.copy(), [cc1.get_contour()], -1, 255, -1)
    img2 = cv.drawContours(blank.copy(), [cc2.get_contour()], -1, 255, -1)

    ccs_intersection = np.logical_and(img1, img2)

    return ccs_intersection.any()


def intersection_percentage(cc1: ConnectedComponent, cc2: ConnectedComponent):
    rects_intersection = intersection(cc1.get_rect(), cc2.get_rect())
    if len(rects_intersection) == 0:
        return 0
    min_area = cc1.get_rect_area()
    min_area = min(min_area, cc2.get_rect_area())
    intersection_area = rects_intersection[2] * rects_intersection[3]

    return intersection_area / min_area


def is_horizontally_aligned_with(a, b):
    y = max(a[1], b[1])
    h = min(a[1] + a[3], b[1] + b[3]) - y
    return h >= 0


def is_vertically_aligned(a, b):
    x = max(a[0], b[0])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    return w >= 0
