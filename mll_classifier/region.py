from typing import List

import numpy as np

from connected_components.connected_components import get_connected_components, set_ccs_neighbors, ConnectedComponent

T_VAR = 1.3


class Region:
    def __init__(self, rect, img, t_var=T_VAR, debug=False):
        super().__init__()
        self.__rect = rect
        self.__root_img = img.copy()
        self.__img = self.__crop_image(img.copy())
        self.__t_var = t_var
        self.__set_all_attrs()
        self.__ccs = get_connected_components(self.__img, (rect[0], rect[1]))
        self.__hcs = set_ccs_neighbors(self.__ccs)
        self.__ccs = [cc for hc in self.__hcs for cc in hc]

    def get_next_level_homogeneous_regions(self, vertical=None):
        region = self
        if vertical is None:
            vertical = self.__var_vw > self.__var_hw

        split = False
        for i in range(2):
            # split = False
            split_regions = []
            if self.get_var_b(vertical) > self.__t_var or self.get_var_w(vertical) > self.__t_var:
                # split2, split_regions = self.__split_region(region, vertical)
                # split = split or split2
                split, split_regions = self.__split_region(region, vertical)

            if split:
                # vertical = not vertical
                # new_split_regions = []
                # for split_region in split_regions:
                #     if split_region.get_var_b(vertical) > self.__t_var or self.get_var_w(vertical) > self.__t_var:
                #         _, split_regions2 = split_region.__split_region(split_region, vertical)
                #         new_split_regions.extend(split_regions2)
                #     else:
                #         new_split_regions.append(split_region)
                # return True, new_split_regions
                return True, split_regions
            elif not split:
                vertical = not vertical

        return False, [region]

    def __split_region(self, region, vertical=False):
        bl = region.get_bl(vertical)
        wl = region.get_wl(vertical)

        if vertical:
            len_bl = [rect[2] for rect in bl]
            len_wl = [rect[2] for rect in wl]
        else:
            len_bl = [rect[3] for rect in bl]
            len_wl = [rect[3] for rect in wl]

        splits = []

        if len(wl) > 0 and region.get_var_w(vertical) > self.__t_var:
            _, black_splits = self.__split_wl(len_wl, wl, vertical)
            splits.extend(black_splits)

        if region.get_var_b(vertical) > self.__t_var:
            _, white_splits = self.__split_bl(len_bl, wl, vertical)
            splits.extend(white_splits)

        if len(splits) == 0:
            return False, []

        regions_rects = self.__get_split_regions_rects(
            region, splits, vertical)

        regions = [Region(region_rect, self.__root_img)
                   for region_rect in regions_rects]

        regions = [region for region in regions if len(region.get_ccs()) > 0]

        return True, regions

    def __split_bl(self, len_bl, wl, vertical):
        median_b = np.median(len_bl)
        max_b = np.max(len_bl)

        for i in range(len(len_bl)):
            if len_bl[i] == max_b and len_bl[i] > median_b:
                if i == 0 or i == len(len_bl) - 1:
                    return self.__split_bl_once(wl, i, vertical)
                else:
                    return self.__split_bl_twice(wl, i, vertical)
        return False, []

    def __split_bl_once(self, wl, i, vertical):
        upper = True
        if i == 0:
            upper = False

        split = self.__get_split(wl, i, upper, vertical)

        return True, [split]

    def __split_bl_twice(self, wl, i, vertical):
        upper_split = self.__get_split(wl, i, True, vertical)
        lower_split = self.__get_split(wl, i, False, vertical)

        return True, [upper_split, lower_split]

    def __split_wl(self, len_wl, wl, vertical):
        median_w = np.median(len_wl)
        max_w = np.max(len_wl)

        for i in range(len(wl)):
            if len_wl[i] == max_w and len_wl[i] > median_w:
                split = self.__get_split(wl, i, False, vertical)

                return True, [split]

        return False, []

    def __get_split_regions_rects(self, region, splits, vertical):
        splits.sort()

        if vertical:
            return self.__get_vertical_split_regions_rects(region, splits)
        else:
            return self.__get_horizontal_split_regions_rects(region, splits)

    def __get_horizontal_split_regions_rects(self, region, splits):
        x, y, w, h = region.get_rect()
        rects = []
        for i, split in enumerate(splits):
            if i == 0:
                rects.append((x, y, w, split - y))
            elif i <= len(splits) - 1:
                rects.append((x, splits[i - 1], w, split - splits[i - 1]))
            if i == len(splits) - 1:
                rects.append((x, split, w, h + y - split))

        return rects

    def __get_vertical_split_regions_rects(self, region, splits):
        x, y, w, h = region.get_rect()
        rects = []
        for i, split in enumerate(splits):
            if i == 0:
                rects.append((x, y, split - x, h))
            elif i <= len(splits) - 1:
                rects.append((splits[i - 1], y, split - splits[i - 1], h))
            if i == len(splits) - 1:
                rects.append((split, y, w + x - split, h))

        return rects

    def __get_split(self, wl, i, upper, vertical):
        start = i
        if upper:
            start -= 1

        x, y, w, h = wl[start]

        if vertical:
            return x + int(w / 2)
        else:
            return y + int(h / 2)

    def __crop_image(self, img):
        h_img, w_img = img.shape[:2]
        x, y, w, h = self.__rect
        if x == 0 and y == 0 and w == w_img and h == h_img:
            return img
        else:
            return img[
                   y:(y + h),
                   x:(x + w)
                   ]

    def __set_all_attrs(self):
        self.__set_projections()
        self.__bl_h, self.__wl_h = self.__get_lines()
        self.__bl_v, self.__wl_v = self.__get_lines(True)
        self.__set_variances()

    def __set_projections(self):
        self.__p_h = np.sum(self.__img, 1, np.uint8) / 255
        self.__p_v = np.sum(self.__img, 0, np.uint8) / 255
        self.__z_h = self.__get_bi_level_projection__(self.__p_h)
        self.__z_v = self.__get_bi_level_projection__(self.__p_v)

    def __get_bi_level_projection__(self, p):
        p = p.copy()
        for i in range(len(p)):
            if p[i] > 0:
                p[i] = 1
            else:
                p[i] = 0
        return p

    def __get_lines(self, vertical=False):
        x, y, w, h = self.__rect
        p = self.__z_h
        if vertical:
            p = self.__z_v

        uppers, lowers = self.__get_bounds__(p, vertical)
        bl = []
        wl = []

        for i in range(len(uppers)):
            if vertical:
                bl.append((uppers[i], y, lowers[i] - uppers[i], h))
            else:
                bl.append((x, uppers[i], w, lowers[i] - uppers[i]))
            if i + 1 < len(uppers):
                if vertical:
                    wl.append(
                        (uppers[i], y, uppers[i + 1] - lowers[i], h))
                else:
                    wl.append(
                        (x, uppers[i], w, uppers[i + 1] - lowers[i]))

        return bl, wl

    def __get_bounds__(self, p, vertical=False):
        th = 0

        x, y, w, h = self.__rect

        length = h
        offset = y
        if vertical:
            length = w
            offset = x

        uppers = []
        lowers = []

        for i in range(length):
            if (i == 0 and p[i] > th) or (i != 0 and p[i] > th >= p[i - 1]):
                uppers.append(i + offset)
            if (i == length - 1 and p[i] > th) or (i != length - 1 and p[i] > th >= p[i + 1]):
                lowers.append(i + offset)

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

    def get_ccs(self) -> List[ConnectedComponent]:
        return self.__ccs

    def set_features(self):
        ccs = self.__ccs

        areas = []
        heights = []
        widths = []
        for cc in ccs:
            if cc.get_area() > 0:
                areas.append(cc.get_area())
            else:
                absolute_area = np.sum(cc.get_contour().flatten()) / 255
                areas.append(absolute_area)
            heights.append(cc.get_rect()[3])
            widths.append(cc.get_rect()[2])

        ws = [cc.get_rnws() for cc in ccs if cc.get_rnws() > 0]

        if len(ws) == 0:
            ws.append(0)

        self.__ws = ws
        self.__max_area = np.max(areas, initial=0)
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

    def set_root_img(self, img):
        self.__init__(self.__rect, img)
        return self

    def get_img(self):
        return self.__img.copy()

    def get_root_img(self):
        return self.__root_img.copy()

    def split_horizontally_at(self, splits):
        if len(splits) == 0:
            return [self]

        regions_rects = self.__get_split_regions_rects(self, splits, False)

        regions = [Region(region_rect, self.__root_img) for region_rect in regions_rects]
        regions = [region for region in regions if len(region.get_ccs()) > 0]

        return regions

    def get_hcs(self) -> List[List[ConnectedComponent]]:
        return self.__hcs.copy()
