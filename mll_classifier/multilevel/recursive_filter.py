import cv2 as cv
import numpy as np


class RecursiveFilter():
    def __init__(self, img, regions):
        super().__init__()
        self.__img = img.copy()
        self.__regions = regions.copy()

    def filter(self):
        for region in self.__regions:
            self.__filter_region(region)

    def __filter_region(self, region):
        features = self.__extract_features(region)
        self.__apply_max_median_filter(region, features)
        self.__apply_min_median_filter(region, features)

    def __extract_features(self, region):
        ccs = self.__get_ccs(region)
        areas = []
        heights = []
        widths = []
        xs = []
        lnws = []
        rnws = []
        n_ln = []
        n_rn = []
        for cc in ccs:
            areas.append(cv.contourArea(cc))
            x, y, w, h = cv.boundingRect(cc)
            heights.append(h)
            widths.append(w)
            xs.append(x)

        for i, x in enumerate(xs):
            lnn_i = x
            rnn_i = x + widths[i]
            for x2 in xs:
                if x2 != x:
                    if x2 < x and (x2 > lnn_i or lnn_i == x):
                        lnn_i = x2
                    elif x2 > x + widths[i] and (x2 < rnn_i or rnn_i == x + widths[i]):
                        rnn_i = x2

            lnws.append(x - lnn_i)
            rnws.append(rnn_i - x)

            num_ln = 0
            if lnn_i != x:
                num_ln += 1
                for x2 in xs:
                    if x2 < lnn_i:
                        num_ln += 1

            num_rn = 0
            if rnn_i != x + widths[i]:
                num_rn += 1
                for x2 in xs:
                    if x2 > rnn_i:
                        num_rn += 1

        ws = []
        ws.extend(lnws)
        ws.extend(rnws)
        ws = [ws_i for ws_i in ws if ws_i > 0]

        features = {
            'ccs': ccs,
            'areas': areas,
            'heights': heights,
            'widths': widths,
            'xs': xs,
            'lnws': lnws,
            'rnws': rnws,
            'n_ln': n_ln,
            'n_rn': n_rn,
            'max_area': np.max(areas),
            'median_area': np.median(areas),
            'mean_area': np.average(areas),
            'max_h': np.max(heights),
            'median_h': np.median(heights),
            'mean_h': np.average(heights),
            'max_w': np.max(widths),
            'median_w': np.median(widths),
            'mean_w': np.average(widths),
            'max_ws': np.max(ws),
            'median_ws': np.median(ws),
            'mean_ws': np.average(ws)
        }

        k = {}
        for feature in 'area', 'h', 'w':
            r = features[f'mean_{feature}'] / features[f'median_{feature}']
            k[feature] = np.max(r, 1 / r)

        features['k'] = k

        return features

    def __apply_max_median_filter(self, region, features):
        k = features['k']

        non_text_sus = []
        for i in range(len(features['ccs'])):
            if features['areas'][i] == features['max_area'] \
                and features['areas'][i] > k['area'] * features['median_area'] \
                    and ((features['heights'][i] == features['max_h']
                          and features['heights'][i] > k['w'] * features['median_h'])
                         or (features['widths'][i] == features['max_w']
                             and features['widths'][i] > k['w'] * features['median_w'])):
                non_text_sus.append(i)

        non_text = self.__classify_non_text(features, non_text_sus)

        return non_text

    def __apply_min_median_filter(self, region, features):
        k = features['k']

        non_text_sus = []
        for i in range(len(features['ccs'])):
            if (features['heights'][i] == features['min_h']
                    and features['heights'][i] < features['median_h'] / k['h']) \
                or (features['widths'][i] == features['min_w']
                    and features['widths'][i] < features['median_w'] / k['w']):
                non_text_sus.append(i)

        non_text = self.__classify_non_text(features, non_text_sus, True)

        return non_text

    def __classify_non_text(self, features, non_text_sus, min=False):
        non_text = []
        for i in non_text_sus:
            is_non_text = (np.min(features['lnws'][i], features['rnws'][i]) > np.max(features['median_ws'], features['mean_ws'])
                           and (np.max(features['lnws'][i], features['rnws'][i]) == features['max_ws']
                                or np.min(features['lnws'][i], features['rnws'][i]) > 2 * features['mean_ws'])) \
                or ((features['n_ln'][i] == np.max(features['n_ln']) and features['n_ln'][i] > 2)
                    or (features['n_rn'][i] == np.max(features['n_rn']) and features['n_rn'][i] > 2))
            if (is_non_text and not min) or (not is_non_text and min):
                non_text.append(features['ccs'][i])

        return non_text

    def __get_ccs(self, region):
        x1, y1, x2, y2 = region
        img = self.__crop_img(x1, y1, x2, y2)
        ccs, _ = cv.findContours(
            img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(x1, y1))[-2:]
        return ccs

    def __crop_img(self, x1, y1, x2, y2):
        return self.__img[y1:y2, x1:x2]
