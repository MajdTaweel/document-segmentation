import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


KERNEL_SIZE = 5
T_VAR = 1.2


class MultilevelClassifier:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()

    def get_horizontal_projection(self):
        regions = [self.__img]

        homogeneous = False
        new_regions = []

        while not homogeneous:
            homogeneous = True
            for region in regions:
                z, var = self.__get_projection_props(region)
                if var > T_VAR:
                    homogeneous = False
                    split_regions = self.__get_regions(z, region)
                    new_regions.extend(split_regions)
                else:
                    z, var = self.__get_projection_props(region, True)
                    if var > T_VAR:
                        homogeneous = False
                        split_regions = self.__get_regions(z, region, True)
                        new_regions.extend(split_regions)
                    else:
                        new_regions.append(region)
            regions = new_regions
            new_regions = []

        for i in range(len(regions)):
            cv.namedWindow(f'Region{i}', cv.WINDOW_FREERATIO)
            cv.imshow(f'Region{i}', regions[i])
            self.__get_projection_props(regions[i], draw=True)

        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

    def __get_projection_props(self, img, vertical=False, draw=False):
        p = self.__get_projection(img, vertical)
        # z = self.__smooth_projection(p, vertical)
        z = self.__get_bi_level_projection(p, vertical).flatten()
        # g = self.__get_projection_derivative(z, vertical)
        # l = self.__get_local_extremes(g)

        # delta = self.__get_delta(l)
        # var = self.__get_variance(delta)
        h, w, uppers, lowers, black_lines, white_lines = self.__get_lines(
            z, img, vertical)
        lines = []
        lines.extend(black_lines)
        lines.extend(white_lines)
        var = np.var(lines)

        lines_props = h, w, uppers, lowers, black_lines, white_lines

        if var > T_VAR:
            if vertical:
                print(f'{var} - VERTICAL')
            else:
                print(f'{var} - HORIZONTAL')

        if var > 1 and draw:
            plt.plot(z, np.arange(0, len(z)), color='black')
            # plt.plot(g, np.arange(0, len(g)), color='blue')
            # plt.plot(np.zeros(len(l)), l, 'r.')
            plt.gca().invert_yaxis()

        plt.show()

        return z, var

    def __get_regions(self, p, region, vertical=False):
        splits, uppers, lowers, w, h = self.__get_splits(p, region, vertical)
        if len(splits) == 0:
            return [region]
        bounding_rects = self.__get_regions_bounding_rects(
            splits, uppers, lowers, w, h, vertical)

        regions = []
        for rect in bounding_rects:
            x, y, x2, y2 = rect
            cropped_region = self.__crop_img(region, x, y, x2, y2)
            if len(cropped_region.flatten()) > 0:
                regions.append(cropped_region)

        return regions

    def __get_splits(self, p, region, vertical=False):
        h, w, uppers, lowers, black_lines, white_lines = self.__get_lines(
            p, region, vertical)

        median_b = np.median(black_lines)
        max_b = np.max(black_lines)

        if len(white_lines) > 0:
            median_w = np.median(white_lines)
            max_w = np.max(white_lines)

        splits = []

        for i in range(len(black_lines)):
            is_splitten = False
            if black_lines[i] > median_b and black_lines[i] == max_b:
                if i != 0 and (len(splits) == 0 or splits[-1] != i - 1):
                    splits.append(i - 1)
                if i < len(white_lines):
                    splits.append(i)

                    is_splitten = True

                    if i + 1 < len(white_lines):
                        median_w = np.median(white_lines[i + 1:])
                        max_w = np.max(white_lines[i + 1:])

                    median_b = np.median(black_lines[i + 1:])
                    max_b = np.max(black_lines[i + 1:])
                continue

            if i < len(white_lines) and not is_splitten and white_lines[i] > median_w and white_lines[i] == max_w:
                splits.append(i)

                if i + 1 < len(white_lines):
                    median_w = np.median(white_lines[i + 1:])
                    max_w = np.max(white_lines[i + 1:])

                median_b = np.median(black_lines[i + 1:])
                max_b = np.max(black_lines[i + 1:])
        return splits, uppers, lowers, w, h

    def __get_regions_bounding_rects(self, splits, uppers, lowers, w, h, vertical=False):
        bounding_rects = []
        if vertical:
            for i in range(len(splits)):
                if i == 0:
                    bounding_rects.append((
                        0,
                        0,
                        lowers[splits[i]] +
                        int((uppers[splits[i] + 1] - lowers[splits[i]]) / 2),
                        h
                    ))
                if i == len(splits) - 1:
                    bounding_rects.append((
                        lowers[splits[i]] +
                        int((uppers[splits[i] + 1] - lowers[splits[i]]) / 2),
                        0,
                        w,
                        h
                    ))
                else:
                    bounding_rects.append((
                        lowers[splits[i]] +
                        int((uppers[splits[i] + 1] - lowers[splits[i]]) / 2),
                        0,
                        lowers[splits[i + 1]] +
                        int((uppers[splits[i + 1] + 1] -
                             lowers[splits[i + 1]]) / 2),
                        h
                    ))
        else:
            for i in range(len(splits)):
                if i == 0:
                    bounding_rects.append((
                        0,
                        0,
                        w,
                        lowers[splits[i]] +
                        int((uppers[splits[i] + 1] - lowers[splits[i]]) / 2)
                    ))
                if i == len(splits) - 1:
                    bounding_rects.append((
                        0,
                        lowers[splits[i]] +
                        int((uppers[splits[i] + 1] - lowers[splits[i]]) / 2),
                        w,
                        h
                    ))
                else:
                    bounding_rects.append((
                        0,
                        lowers[splits[i]] +
                        int((uppers[splits[i] + 1] - lowers[splits[i]]) / 2),
                        w,
                        lowers[splits[i + 1]] +
                        int((uppers[splits[i + 1] + 1] -
                             lowers[splits[i + 1]]) / 2)
                    ))
        return bounding_rects

    def __get_lines(self, p, region, vertical=False):
        th = 0
        h, w = region.shape[:2]

        length = h
        if vertical:
            length = w

        uppers = [i for i in range(length - 1) if p[i] <= th and p[i + 1] > th]
        lowers = [i for i in range(length - 1) if p[i] > th and p[i + 1] <= th]

        if len(uppers) < len(lowers) or len(uppers) == 0:
            uppers.insert(0, 0)
        if len(lowers) < len(uppers) or len(lowers) == 0:
            lowers.append(length)

        black_lines = []
        white_lines = []

        for i in range(len(uppers)):
            black_lines.append(lowers[i] - uppers[i])
            if i + 1 < len(uppers):
                white_lines.append(uppers[i + 1] - lowers[i])

        return h, w, uppers, lowers, black_lines, white_lines

    def __crop_img(self, img, x, y, x2, y2):
        return img[y:y2, x:x2]
        # cropped_img = img[y:y2, x:x2]
        # coords = cv.findNonZero(cropped_img)
        # x_cropped, y_cropped, w_cropped, h_cropped = cv.boundingRect(coords)
        # return cropped_img[y_cropped:y_cropped + h_cropped, x_cropped: x_cropped + w_cropped]

    def __get_projection(self, img, vertical=False):
        axis = 1
        if vertical:
            axis = 0
        return cv.reduce(img, axis, cv.REDUCE_SUM, dtype=cv.CV_32S) / 255

    def __smooth_projection(self, p, vertical=False, kernel_size=KERNEL_SIZE):
        kernel = (1, kernel_size * 2)
        if vertical:
            kernel = (kernel_size * 2, 1)

        return np.floor(cv.blur(p, kernel))

    def __get_bi_level_projection(self, p, vertical=False):
        p = p.copy()
        if vertical:
            for i in range(len(p[0])):
                if p[0][i] > 0:
                    p[0][i] = 1
                else:
                    p[0][i] = 0
        else:
            for i in range(len(p)):
                if p[i][0] > 0:
                    p[i][0] = 1
                else:
                    p[i][0] = 0
        return p

    def __get_projection_derivative(self, z, vertical=False):
        dx = 0
        if vertical:
            dx = 1
        return cv.Scharr(z, cv.CV_64F, dx, 1 - dx)
        # return cv.Sobel(z, cv.CV_64F, dx, 1 - dx, ksize=5)

    def __get_local_extremes(self, g):
        l = []
        g_flat = g.flatten()
        for i in range(len(g_flat) - 1):
            if (g_flat[i] < 0 and g_flat[i + 1] >= 0) or (g_flat[i] > 0 and g_flat[i + 1] <= 0):
                l.append(i)
        return l

    def __get_delta(self, l):
        delta = []
        for i in range(len(l) - 1):
            delta.append(l[i + 1] - l[i])
        return delta

    def __get_variance(self, delta):
        return np.var(delta)
