import sys

import cv2 as cv
import numpy as np

from heuristic_filter.heuristic_filter import HeuristicFilter
from mll_classifier.mll_classifier import MllClassifier
from postprocessor.postprocessor import Postprocessor
from preprocessor.preprocessor import Preprocessor
from text_segmenter.text_segmenter import TextSegmenter


def main(path):
    src = cv.imread(path, cv.IMREAD_UNCHANGED)

    preprocessed = Preprocessor(src).preprocess()

    ccs_text, ccs_non_text, img_text = HeuristicFilter(preprocessed).filter()

    ccs_text, ccs_non_text2, img_text = MllClassifier(
        img_text).classify_non_text_ccs()

    ccs_non_text.extend(ccs_non_text2)

    ccs_text, ccs_non_text, img_text = Postprocessor(
        preprocessed, ccs_text, ccs_non_text).postprocess()

    ccs_text = TextSegmenter(img_text, ccs_text, ccs_non_text).segment_text()

    cv.drawContours(src, [cc.get_contour()
                          for cc in ccs_text], -1, (0, 255, 0), 2)
    cv.drawContours(src, [cc.get_contour()
                          for cc in ccs_non_text], -1, (0, 0, 255), 2)
    cv.namedWindow('Contours', cv.WINDOW_FREERATIO)
    cv.imshow('Contours', src)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


# def get_bounding_rect(cc):
#     x, y, w, h = cv.boundingRect(cc)

#     Rotated rect
#     rect = cv.minAreaRect(cnt)
#     box = cv.boxPoints(rect)
#     box = np.int0(box)
#     cv.drawContours(img,[box],0,(0,0,255),2)

#     return x, y, w, h

if __name__ == '__main__':
    # args = sys.argv[1:]
    # if len(args) == 0:
    #     print('Argument missing. Taking argument from input:')
    #     args.append(input())
    # elif len(args) > 1:
    #     raise Exception(f'Too many arguments: {len(args)}. Only one argument is required.')

    # main(args[0])
    main('img/man.jpg')
