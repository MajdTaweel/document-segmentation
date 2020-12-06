import cv2 as cv

from heuristic_filter.heuristic_filter import HeuristicFilter
from mll_classifier.mll_classifier import MllClassifier
from preprocessor.preprocessor import Preprocessor
from text_segmenter.text_segmenter import TextSegmenter
from non_text_classifier.non_text_classifier import NonTextClassifier
from region_refiner.region_refiner import RegionRefiner


def show_and_wait(title, img):
    cv.namedWindow(title, cv.WINDOW_FREERATIO)
    cv.imshow(title, img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def draw_contours_then_show_and_wait(title, img, ccs_and_colors_list):
    img = img.copy()
    for ccs_and_color in ccs_and_colors_list:
        cv.drawContours(img, [cc.get_contour() for cc in ccs_and_color[0]], -1, ccs_and_color[1], 2)
    show_and_wait(title, img)


class DocumentAnalyzer:
    def __init__(self, path, debug=False):
        self.__src = cv.imread(path, cv.IMREAD_UNCHANGED)
        self.__preprocessor = Preprocessor(self.__src)
        self.__region_refiner = RegionRefiner()
        self.__img_resized = None
        self.__img_text = None
        self.__ccs_text = None
        self.__ccs_non_text = None
        self.__ccs_dict = None
        self.__img_labeled = None
        self.__img_labeled_original_size = None
        self.__debug = debug

    def analyze_document(self):
        if self.__debug:
            show_and_wait('Image', self.__src)

        preprocessed = self.__preprocess()
        self.__apply_heuristic_filter(preprocessed)
        self.__apply_mll_classifier()
        self.__segment_text()
        self.__classify_non_text_element()
        self.__label_regions()
        self.__rescale_img_to_original()

    def __preprocess(self):
        self.__img_resized = self.__preprocessor.get_resized_img()
        preprocessed = self.__preprocessor.preprocess()

        if self.__debug:
            show_and_wait('Preprocessed', preprocessed)

        return preprocessed

    def __apply_heuristic_filter(self, preprocessed):
        self.__ccs_text, self.__ccs_non_text, self.__img_text = HeuristicFilter(preprocessed).filter()

        if self.__debug:
            ccs_and_colors = [
                (self.__ccs_text, (0, 255, 0)),
                (self.__ccs_non_text, (0, 0, 255))
            ]
            draw_contours_then_show_and_wait('Heuristic Filter', self.__img_resized, ccs_and_colors)

    def __apply_mll_classifier(self):
        self.__ccs_text, mll_ccs_non_text, img_text = MllClassifier(self.__img_text).classify_non_text_ccs()
        self.__ccs_non_text.extend(mll_ccs_non_text)

        if self.__debug:
            ccs_and_colors = [
                (self.__ccs_text, (0, 255, 0)),
                (self.__ccs_non_text, (0, 0, 255))
            ]
            draw_contours_then_show_and_wait('MLL Classifier', self.__img_resized, ccs_and_colors)

    def __segment_text(self):
        self.__ccs_text, self.__ccs_non_text = TextSegmenter(self.__img_text, self.__ccs_text, self.__ccs_non_text,
                                                             self.__img_resized).segment_text()

        if self.__debug:
            ccs_and_colors = [
                (self.__ccs_text, (0, 255, 0)),
                (self.__ccs_non_text, (0, 0, 255))
            ]
            draw_contours_then_show_and_wait('Segmented', self.__img_resized, ccs_and_colors)

    def __classify_non_text_element(self):
        self.__ccs_dict = NonTextClassifier(self.__img_resized.shape[:2], self.__ccs_text,
                                            self.__ccs_non_text).classify_non_text_elements()

    def __label_regions(self):
        self.__img_labeled = self.__region_refiner.label_regions(self.__img_resized, self.__ccs_dict)

        if self.__debug:
            show_and_wait('Labeled', self.__img_labeled)

    def __rescale_img_to_original(self):
        self.__img_labeled_original_size = self.__preprocessor.resize_img_to_original_size(self.__img_labeled)
