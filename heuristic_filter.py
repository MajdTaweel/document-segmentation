import cv2 as cv


T_AREA = 6
T_INSIDE = 4


class HeuristicFilter:
    def __init__(self, img, t_area=T_AREA, t_inside=T_INSIDE):
        super().__init__()
        self.__img = img.copy()
        self.__t_inside = t_inside
        self.__t_area = t_area

    def filter(self):
        """
        Heuristic filter.

        Filters out not-text connected components (CCs) from the input ccs.

        Parameters:
            css: CCs (contours) of an image

        Returns:
            ccs_text: CCs of text components
            ccs_img: CCs of non_text components
        """

        ccs_noise = self.__get_ccs_noise()

        ccs_denoise = self.__filter_noise(ccs_noise)

        ccs_text, ccs_non_text = self.__filter_non_text(ccs_denoise)

        return ccs_text, ccs_non_text

    def __get_ccs(self):
        ccs, hierarchy = cv.findContours(
            self.__img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return ccs

    def __get_ccs_noise(self):
        ccs = self.__get_ccs()
        ccs_noise = []
        for i, cc in enumerate(ccs):
            area = cv.contourArea(cc)
            if area < self.__t_area:
                ccs_noise.append(cc)
                continue

        return ccs_noise

    def __filter_noise(self, ccs_noise):
        cv.drawContours(self.__img, ccs_noise, -1, (255, 255, 255), -1)
        ccs = self.__get_ccs()
        return ccs

    def __filter_non_text(self, ccs):
        ccs_non_text = []
        for i, cc in enumerate(ccs):
            if self.__has_descendants_more_than_t_inside(ccs, i):
                ccs_non_text.append(cc)
                x, y, w, h = cv.boundingRect(cc)
                cv.rectangle(self.__img, (x, y), (x + w, y + h),
                             (255, 255, 255), -1)

        ccs_text = self.__get_ccs()

        return ccs_text, ccs_non_text

    def __has_descendants_more_than_t_inside(self, ccs, i):
        num_descendants = 0
        x_i, y_i, w_i, h_i = cv.boundingRect(ccs[i])
        for j, cc_j in enumerate(ccs, start=i + 1):
            x_j, y_j, w_j, h_j = cv.boundingRect(cc_j)
            if x_j >= x_i and y_j >= y_i and x_j + w_j <= x_i + w_i and y_j + h_j <= y_i + h_i:
                num_descendants += 1
                if num_descendants > self.__t_inside:
                    return True

        return False
