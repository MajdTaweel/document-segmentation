import cv2 as cv
import numpy as np


class RecursiveFilter():
    def __init__(self, img, region):
        super().__init__()
        self.__img = img.copy()
        # self.__regions = regions.copy()
        self.__region = region

    # def filter(self):
    #     modified = False
    #     non_text = []
    #     for region in self.__regions:
    #         current_modified, non_text_i = self.__filter_region(region)
    #         modified = modified or current_modified
    #         non_text.extend(non_text_i)

    #     return modified, self.__img, non_text

    def filter(self):
        return self.__filter_region(self.__region)

    def __filter_region(self, region):
        region.set_features()

        non_text_max = self.__apply_max_median_filter(region)
        non_text_min = self.__apply_min_median_filter(region)

        non_text = []
        non_text.extend(non_text_max)
        non_text.extend(non_text_min)

        cv.drawContours(self.__img, [cc.get_contour()
                                     for cc in non_text], -1, (0, 0, 0), -1)

        region.set_img(self.__img)

        return len(non_text) > 0, self.__img, non_text

    def __apply_max_median_filter(self, region):
        features = region.get_features()
        ccs = region.get_ccs()
        k = features['k']

        non_text_sus = []
        for cc in ccs:
            w = cc.get_rect()[2]
            h = cc.get_rect()[3]
            if cc.get_area() == features['max_area'] \
                and cc.get_area() > k['area'] * features['median_area'] \
                    and ((h == features['max_h'] and h > k['w'] * features['median_h'])
                         or (w == features['max_w'] and w > k['w'] * features['median_w'])):
                non_text_sus.append(cc)

        return self.__classify_non_text(region, non_text_sus)

    def __apply_min_median_filter(self, region):
        features = region.get_features()
        ccs = region.get_ccs()
        k = features['k']

        non_text_sus = []
        for cc in ccs:
            w = cc.get_rect()[2]
            h = cc.get_rect()[3]
            if (h == features['min_h'] and h < features['median_h'] / k['h']) \
                    or (w == features['min_w'] and w < features['median_w'] / k['w']):
                non_text_sus.append(cc)

        return self.__classify_non_text(region, non_text_sus)

    def __classify_non_text(self, region, non_text_sus):
        features = region.get_features()
        non_text = []
        for cc in non_text_sus:
            lnws = cc.get_lnws()
            rnws = cc.get_rnws()
            num_ln = cc.get_num_ln()
            num_rn = cc.get_num_rn()
            if (np.min([lnws, rnws]) > np.max([features['median_ws'], features['mean_ws']])
                    and (np.max([lnws, rnws]) == features['max_ws'] or np.min([lnws, rnws]) > 2 * features['mean_ws'])) \
                    or ((num_ln == np.max(features['n_ln']) and num_ln > 2)
                        or (num_rn == np.max(features['n_rn']) and num_rn > 2)):
                non_text.append(cc)

        return non_text
