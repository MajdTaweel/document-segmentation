import sys
import cv2 as cv
import numpy as np
from preprocessor.preprocessor import Preprocessor
from heuristic_filter.heuristic_filter import HeuristicFilter
from mll_classifier.multilevel.multilevel_classifier import MultilevelClassifier


def main(path):
    src = cv.imread(path, cv.IMREAD_UNCHANGED)

    preprocessed = Preprocessor(src).preprocess()

    ccs_text, ccs_non_text = HeuristicFilter(preprocessed).filter()

    # con_img = cv.drawContours(src, ccs_text, -1, (0, 255, 0), 2)

    con_img = cv.drawContours(src, ccs_non_text, -1, (0, 0, 255), 2)

    # img_text = cv.drawContours(preprocessed, ccs_non_text, -1, (0, 0, 0), -1)
    img_text = cv.drawContours(
        np.zeros(preprocessed.shape, dtype=np.uint8), ccs_text, -1, (255, 255, 255), -1)

    # cv.namedWindow('Contours', cv.WINDOW_FREERATIO)
    # cv.imshow('Contours', con_img)

    MultilevelClassifier(img_text).get_horizontal_projection()

    # if cv.waitKey(0) & 0xff == 27:
    #     cv.destroyAllWindows()


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
    main('img/la.png')
