import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


KERNEL_SIZE = 5
T_VAR = 1.3


class MultilevelClassifier:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()

    def get_horizontal_projection(self):
        self.__get_homogeneous_regions()

    def __get_homogeneous_regions(self):
        h, w = self.__img.shape[:2]
        regions = [(0, 0, w, h)]
        dirs = [True]

        i = 0
        no_split = False
        while i < len(regions):
            split = False
            split_regions = []
            props = self.__get_projection_props(regions[i], dirs[i])
            if props['var_b'] == T_VAR or props['var_w'] > T_VAR:
                split, split_regions = self.__split_region(
                    regions[i], props, dirs[i])
                if not split and not no_split:
                    dirs[i] = not dirs[i]
                    no_split = True
                    continue
            elif not no_split:
                dirs[i] = not dirs[i]
                no_split = True
                continue

            if split:
                for j, split_region in enumerate(split_regions):
                    regions.insert(i + j + 1, split_region)
                    dirs.insert(i + j + 1, not dirs[i])
                regions.pop(i)
                dirs.pop(i)
            else:
                i += 1

            no_split = False

        dirs.clear()

        print(len(regions))

        for i in range(len(regions)):
            x1, y1, x2, y2 = regions[i]
            img = self.__crop_img(x1, y1, x2, y2)
            # cv.namedWindow(f'Region{i}', cv.WINDOW_FREERATIO)
            cv.imshow(f'Region{i}', img)

        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

    def __get_projection_props(self, region, vertical=False):
        x1, y1, x2, y2 = region
        img = self.__img
        h, w = self.__img.shape[:2]
        if region[0] != 0 or region[1] != 0 or region[2] != w or region[3] != h:
            img = self.__crop_img(x1, y1, x2, y2)

        p = self.__get_projection(img, vertical)
        z = self.__get_bi_level_projection(p, vertical).flatten()

        offset = y1
        if vertical:
            offset = x1
        uppers, lowers, black_lines, white_lines = self.__get_lines(
            img, z, offset, vertical)

        var_b = 0
        if len(black_lines):
            var_b = np.var(black_lines)

        var_w = 0
        if len(white_lines):
            var_w = np.var(white_lines)

        props = {
            'z': z,
            'uppers': uppers,
            'lowers': lowers,
            'black_lines': black_lines,
            'white_lines': white_lines,
            'var_b': var_b,
            'var_w': var_w
        }

        if var_b > T_VAR:
            if vertical:
                print(f'Black variance (Vertical): {var_b}')
            else:
                print(f'Black variance (Horizontal): {var_b}')
        elif var_w > T_VAR:
            if vertical:
                print(f'White variance (Vertical): {var_w}')
            else:
                print(f'White variance (Horizontal): {var_w}')

        return props

    def __split_region(self, region, props, vertical=False):

        median_b = np.median(props['black_lines'])
        max_b = np.max(props['black_lines'])

        if len(props['white_lines']) > 0:
            median_w = np.median(props['white_lines'])
            max_w = np.max(props['white_lines'])

        x1, y1, x2, y2 = region

        if props['var_b'] > props['var_w']:
            for i in range(len(props['black_lines'])):
                if props['black_lines'][i] == max_b and props['black_lines'][i] > median_b:
                    if i == 0 or i == len(props['black_lines']) - 1:
                        split = 0
                        if i == 0:
                            split = props['lowers'][i] + \
                                int((props['uppers'][i + 1] -
                                     props['lowers'][i]) / 2)
                        elif i == len(props['black_lines']) - 1:
                            split = props['lowers'][i - 1] + \
                                int((props['uppers'][i] -
                                     props['lowers'][i - 1]) / 2)
                        if vertical:
                            return True, [(x1, y1, split, y2), (split, y1, x2, y2)]
                        else:
                            return True, [(x1, y1, x2, split), (x1, split, x2, y2)]
                    else:
                        split1 = props['lowers'][i - 1] + \
                            int((props['uppers'][i] -
                                 props['lowers'][i - 1]) / 2)
                        split2 = props['lowers'][i] + \
                            int((props['uppers'][i + 1] -
                                 props['lowers'][i]) / 2)
                        if vertical:
                            return True, [
                                (x1, y1, split1, y2),
                                (split1, y1, split2, y2),
                                (split2, y1, x2, y2)
                            ]
                        else:
                            return True, [
                                (x1, y1, x2, split1),
                                (x1, split1, x2, split2),
                                (x1, split2, x2, y2)
                            ]
        elif len(props['white_lines']) > 0:
            for i in range(len(props['white_lines'])):
                if props['white_lines'][i] == max_w and props['white_lines'][i] > median_w:
                    split = props['lowers'][i] + \
                        int((props['uppers'][i + 1] - props['lowers'][i]) / 2)
                    if vertical:
                        return True, [(x1, y1, split, y2), (split, y1, x2, y2)]
                    else:
                        return True, [(x1, y1, x2, split), (x1, split, x2, y2)]

        return False, []

    def __get_lines(self, region, p, offset, vertical=False):
        th = 0
        h, w = region.shape[:2]

        length = h
        if vertical:
            length = w

        uppers = [i + offset for i in range(length) if (
            i == 0 and p[i] > th) or (i != 0 and p[i] > th and p[i - 1] <= th)]
        lowers = [i + offset for i in range(length) if (i == length - 1 and p[i] > th) or (
            i != length - 1 and p[i] > th and p[i + 1] <= th)]

        black_lines = []
        white_lines = []

        for i in range(len(uppers)):
            black_lines.append(lowers[i] - uppers[i])
            if i + 1 < len(uppers):
                white_lines.append(uppers[i + 1] - lowers[i])

        return uppers, lowers, black_lines, white_lines

    def __crop_img(self, x1, y1, x2, y2):
        return self.__img[y1:y2, x1:x2]
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
