import cv2 as cv
import numpy as np
from connected_components.connected_components import get_connected_components

T_VAR = 1.3


class Region:
    def __init__(self, rect, img, t_var=T_VAR):
        super().__init__()
        self.__rect = rect
        self.__root_img = img
        self.__img = self.__crop_image(img)
        self.__t_var = t_var
        self.__set_all_attrs()
        self.__ccs = get_connected_components(self.__img, (rect[0], rect[1]))

    def get_next_level_homogeneous_regions(self):
        i = 0
        regions = [self]
        vertical = self.__var_vw > self.__var_hw
        no_split = False
        while i < len(regions):
            split = False
            split_regions = []
            if self.get_var_b(vertical) == self.__t_var or self.get_var_w(vertical) > self.__t_var:
                split, split_regions = self.__split_region(
                    regions[i], vertical)

            if split:
                for j, split_region in enumerate(split_regions):
                    regions.insert(i + j + 1, split_region)
                regions.pop(i)
                i += len(split_regions) - 1
            elif not no_split and len(split_regions) < 2:
                vertical = not vertical
                no_split = True
                continue
            else:
                i += 1

            no_split = False

        return len(regions) > 1, regions

    def __split_region(self, region, vertical=False):
        bl = region.get_bl(vertical)
        wl = region.get_wl(vertical)

        len_bl = []
        len_wl = []

        if vertical:
            len_bl = [rect[2] for rect in bl]
            len_wl = [rect[2] for rect in wl]
        else:
            len_bl = [rect[3] for rect in bl]
            len_wl = [rect[3] for rect in wl]

        is_split = False
        region_splits = []

        if region.get_var_b(vertical) > region.get_var_w(vertical):
            is_split, region_splits = self.__split_bl(
                region, len_bl, wl, vertical)
        elif len(wl) > 0:
            is_split, region_splits = self.__split_wl(
                region, len_wl, wl, vertical)

        return is_split, [Region(region_split, self.__root_img) for region_split in region_splits]

    def __split_bl(self, region, len_bl, wl, vertical):
        median_b = np.median(len_bl)
        max_b = np.max(len_bl)

        for i in range(len(len_bl)):
            if len_bl[i] == max_b and len_bl[i] > median_b:
                if i == 0 or i == len(len_bl) - 1:
                    return self.__split_bl_once(region, wl, i, vertical)
                else:
                    return self.__split_bl_twice(region, wl, i, vertical)
        return False, []

    def __split_bl_once(self, region, wl, i, vertical):
        split_type = 'upper'
        if i == 0:
            split_type = 'lower'

        split_regions_rects = self.__get_split_regions_rects(
            region, wl, i, split_type, vertical)

        return True, split_regions_rects

    def __split_bl_twice(self, region, wl, i, vertical):
        split_regions_rects = self.__get_split_regions_rects(
            region, wl, i, 'both', vertical)

        return True, split_regions_rects

    def __split_wl(self, region, len_wl, wl, vertical):
        median_w = np.median(len_wl)
        max_w = np.max(len_wl)

        for i in range(len(wl)):
            if len_wl[i] == max_w and len_wl[i] > median_w:
                split_regions_rects = self.__get_split_regions_rects(
                    region, wl, i, 'lower', vertical)

                return True, split_regions_rects

        return False, []

    def __get_split_regions_rects(self, region, wl, i, split_type, vertical=False):
        x, y, w, h = region.get_rect()

        splits = {}
        if split_type == 'upper' or split_type == 'both':
            splits['upper'] = self.__get_split(wl, i, True, vertical)
        if split_type == 'lower' or split_type == 'both':
            splits['lower'] = self.__get_split(wl, i, False, vertical)

        if split_type == 'both':
            if vertical:
                return [
                    (x, y, splits['upper'] - x, h),
                    (splits['upper'], y, splits['lower'] - splits['upper'], h),
                    (splits['lower'], y, w + x - splits['lower'], h)
                ]
            else:
                return [
                    (x, y, w, splits['upper'] - y),
                    (x, splits['upper'], w, splits['lower'] - splits['upper']),
                    (x, splits['lower'], w, h + y - splits['lower'])
                ]
        else:
            if vertical:
                return [(x, y, splits[split_type] - x, h), (splits[split_type], y, w + x - splits[split_type], h)]
            else:
                return [(x, y, w, splits[split_type] - y), (x, splits[split_type], w, h + y - splits[split_type])]

    def __get_split(self, wl, i, upper, vertical):
        start = i
        if upper:
            start -= 1

        if vertical:
            return wl[start][0] + int(wl[start][2] / 2)
        else:
            return wl[start][1] + int(wl[start][3] / 2)

    def __crop_image(self, img):
        h, w = img.shape[:2]
        if self.__rect[0] == 0 and self.__rect[1] == 0 and self.__rect[2] == w and self.__rect[3] == h:
            return img.copy()
        else:
            return img[
                self.__rect[1]:self.__rect[1] + self.__rect[3],
                self.__rect[0]:self.__rect[0] + self.__rect[2]
            ]

    def __set_all_attrs(self):
        self.__set_projections()
        self.__bl_h, self.__wl_h = self.__get_lines()
        self.__bl_v, self.__wl_v = self.__get_lines(True)
        self.__set_variances()

    def __set_projections(self):
        self.__p_h = cv.reduce(
            self.__img, 1, cv.REDUCE_SUM, dtype=cv.CV_32S) / 255
        self.__p_v = cv.reduce(
            self.__img, 0, cv.REDUCE_SUM, dtype=cv.CV_32S) / 255
        self.__z_h = self.__get_bi_level_projection__(self.__p_h)
        self.__z_v = self.__get_bi_level_projection__(self.__p_v, True)

    def __get_bi_level_projection__(self, p, vertical=False):
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
        return p.flatten()

    def __get_lines(self, vertical=False):
        p = self.__z_h
        length = self.__rect[3]
        if vertical:
            length = self.__rect[2]
            p = self.__z_v

        uppers, lowers = self.__get_bounds__(p, vertical)
        bl = []
        wl = []

        for i in range(len(uppers)):
            if vertical:
                bl.append((uppers[i], 0, lowers[i] - uppers[i], length))
            else:
                bl.append((0, uppers[i], length, lowers[i] - uppers[i]))
            if i + 1 < len(uppers):
                if vertical:
                    wl.append(
                        (uppers[i], 0, uppers[i + 1] - lowers[i], length))
                else:
                    wl.append(
                        (0, uppers[i], length, uppers[i + 1] - lowers[i]))

        return bl, wl

    def __get_bounds__(self, p, vertical=False):
        th = 0

        length = self.__rect[3]
        offset = self.__rect[1]
        if vertical:
            length = self.__rect[2]
            offset = self.__rect[0]

        uppers = [i + offset for i in range(length) if (i == 0 and p[i] > th)
                  or (i != 0 and p[i] > th and p[i - 1] <= th)]
        lowers = [i + offset for i in range(length) if (i == length - 1 and p[i] > th) or (
            i != length - 1 and p[i] > th and p[i + 1] <= th)]

        return uppers, lowers

    def __set_variances(self):
        self.__var_hb, self.__var_hw, self.__var_vb, self.__var_vw = (
            0, 0, 0, 0)

        if len(self.__bl_h) > 0:
            self.__var_hb = np.var([rect[3] for rect in self.__bl_h])

        if len(self.__wl_h) > 0:
            self.__var_hw = np.var([rect[3] for rect in self.__wl_h])

        if len(self.__bl_v) > 0:
            self.__var_vb = np.var([rect[2] for rect in self.__bl_v])

        if len(self.__wl_v) > 0:
            self.__var_vw = np.var([rect[2] for rect in self.__wl_v])

    def get_rect(self):
        return self.__rect

    def get_bl_h(self):
        return self.__bl_h

    def get_wl_h(self):
        return self.__wl_h

    def get_bl_v(self):
        return self.__bl_v

    def get_wl_v(self):
        return self.__wl_v

    def get_bl(self, vertical):
        if vertical:
            return self.__bl_v
        else:
            return self.__bl_h

    def get_wl(self, vertical):
        if vertical:
            return self.__wl_v
        else:
            return self.__wl_h

    def get_var_hb(self):
        return self.__var_hb

    def get_var_hw(self):
        return self.__var_hw

    def get_var_vb(self):
        return self.__var_vb

    def get_var_vw(self):
        return self.__var_vw

    def get_var_b(self, vertical):
        if vertical:
            return self.__var_vb
        else:
            return self.__var_hb

    def get_var_w(self, vertical):
        if vertical:
            return self.__var_vw
        else:
            return self.__var_hw

    def get_ccs(self):
        return self.__ccs

    def set_features(self):
        ccs = self.get_ccs()

        areas = []
        heights = []
        widths = []
        for cc in ccs:
            if cc.get_area() > 0:
                areas.append(cc.get_area())
            else:
                areas.append(cc.get_rect_area())
            heights.append(cc.get_rect()[3])
            widths.append(cc.get_rect()[2])

        ws = [cc.get_rnws()
              for cc in ccs if cc.set_features(ccs).get_rnws() > 0]

        if len(ws) == 0:
            ws.append(0)

        self.__ws = ws
        self.__max_area = np.max(areas)
        self.__median_area = np.median(areas)
        self.__mean_area = np.average(areas)
        self.__max_h = np.max(heights)
        self.__min_h = np.min(heights)
        self.__median_h = np.median(heights)
        self.__mean_h = np.average(heights)
        self.__max_w = np.max(widths)
        self.__min_w = np.min(widths)
        self.__median_w = np.median(widths)
        self.__mean_w = np.average(widths)
        self.__max_ws = np.max(ws)
        self.__median_ws = np.median(ws)
        self.__mean_ws = np.average(ws)
        self.__n_ln = [cc.get_num_ln() for cc in ccs]
        self.__n_rn = [cc.get_num_rn() for cc in ccs]

        r = self.__mean_area / self.__median_area
        self.__k_area = np.max([r, 1 / r])

        r = self.__mean_h / self.__median_h
        self.__k_h = np.max([r, 1 / r])

        r = self.__mean_w / self.__median_w
        self.__k_w = np.max([r, 1 / r])
        return self

    def get_features(self):
        return {
            'ws': self.__ws,
            'max_area': self.__max_area,
            'median_area': self.__median_area,
            'mean_area': self.__mean_area,
            'max_h': self.__max_h,
            'min_h': self.__min_h,
            'median_h': self.__median_h,
            'mean_h': self.__mean_h,
            'max_w': self.__max_w,
            'min_w': self.__min_w,
            'median_w': self.__median_w,
            'mean_w': self.__mean_w,
            'max_ws': self.__max_ws,
            'median_ws': self.__median_ws,
            'mean_ws': self.__mean_ws,
            'n_ln': self.__n_ln,
            'n_rn': self.__n_rn,
            'k': {
                'area': self.__k_area,
                'h': self.__k_h,
                'w': self.__k_w
            }
        }

    def set_img(self, img):
        self.__init__(self.__rect, img)
        return self

    def get_img(self):
        return self.__img
