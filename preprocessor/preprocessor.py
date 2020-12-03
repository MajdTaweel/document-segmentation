import cv2 as cv

from .binarizer import Binarizer

KERNEL_SIZE = 5
# Aspect ration of 1.414:1 is preferred which is A4 paper's aspect ratio
SCALE_RESOLUTION = (850, 1100)
SCALE_RESOLUTION_INV = (1100, 850)


class Preprocessor:
    def __init__(self, img):
        super().__init__()
        self.__img = img.copy()
        self.__original_img_size = img.shape[:2]
        self.__resized_img = self.__resize_img(self.__img)

    def preprocess(self):
        self.__img_to_gray_scale()
        # self.__enhance_img()
        self.__smooth_img()
        binarizer = Binarizer(self.__img)
        self.__img = binarizer.binarize()
        # self.__correct_skew()
        self.__img = self.__resize_img(self.__img)
        return self.__img

    def __resize_img(self, img):
        if img.shape[0] > img.shape[1]:
            return cv.resize(img.copy(), SCALE_RESOLUTION, interpolation=cv.INTER_AREA)
        else:
            return cv.resize(img.copy(), SCALE_RESOLUTION_INV, interpolation=cv.INTER_AREA)

    def __img_to_gray_scale(self):
        if len(self.__img.shape) == 3:
            self.__img = cv.cvtColor(self.__img, cv.COLOR_BGR2GRAY)

    def __enhance_img(self):
        self.__img = cv.equalizeHist(self.__img)

    def __smooth_img(self):
        # self.__img = cv.GaussianBlur(self.__img, (KERNEL_SIZE, KERNEL_SIZE), 0)
        # self.__img = cv.medianBlur(self.__img, KERNEL_SIZE)
        # kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.float32)
        # kernel /= (KERNEL_SIZE ** 2)
        # self.__img = cv.filter2D(self.__img, -1, kernel)
        self.__img = cv.blur(self.__img, (KERNEL_SIZE, KERNEL_SIZE))

    def __correct_skew(self):
        pts = cv.findNonZero(self.__img)
        ret = cv.minAreaRect(pts)

        (cx, cy), (w, h), ang = ret
        if w > h:
            w, h = h, w
            ang += 90

        m = cv.getRotationMatrix2D((cx, cy), ang, 1.0)
        self.__img = cv.warpAffine(
            self.__img, m, (self.__img.shape[1], self.__img.shape[0]))

    def get_resized_img(self):
        return self.__resized_img

    def resize_img_to_original_size(self, img):
        return cv.resize(img, self.__original_img_size, interpolation=cv.INTER_AREA)
