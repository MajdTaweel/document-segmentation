import cv2 as cv

from heuristic_filter.heuristic_filter import HeuristicFilter
from mll_classifier.mll_classifier import MllClassifier
from preprocessor.preprocessor import Preprocessor
from text_segmenter.text_segmenter import TextSegmenter
from non_text_classifier.non_text_classifier import NonTextClassifier
from region_refiner.region_refiner import RegionRefiner


def main(path):
    src = cv.imread(path, cv.IMREAD_UNCHANGED)

    cv.namedWindow('Image', cv.WINDOW_FREERATIO)
    cv.imshow('Image', src)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # PREPROCESSOR ##########################

    preprocessor = Preprocessor(src)
    resized_img = preprocessor.get_resized_img()
    preprocessed = preprocessor.preprocess()

    cv.namedWindow('Preprocessed', cv.WINDOW_FREERATIO)
    cv.imshow('Preprocessed', preprocessed)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # HEURISTIC FILTER ##########################

    ccs_text, ccs_non_text, img_text = HeuristicFilter(preprocessed).filter()

    img_h_filter = resized_img.copy()
    cv.drawContours(img_h_filter, [cc.get_contour()
                                   for cc in ccs_text], -1, (0, 255, 0), 2)
    cv.drawContours(img_h_filter, [cc.get_contour()
                                   for cc in ccs_non_text], -1, (0, 0, 255), 2)

    cv.namedWindow('Heuristic Filter', cv.WINDOW_FREERATIO)
    cv.imshow('Heuristic Filter', img_h_filter)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # MLL CLASSIFIER ##########################

    ccs_text, mll_ccs_non_text, img_text = MllClassifier(img_text).classify_non_text_ccs()
    ccs_non_text.extend(mll_ccs_non_text)

    img_mll = resized_img.copy()
    cv.drawContours(img_mll, [cc.get_contour()
                              for cc in ccs_text], -1, (0, 255, 0), 2)
    cv.drawContours(img_mll, [cc.get_contour()
                              for cc in ccs_non_text], -1, (0, 0, 255), 2)

    cv.namedWindow('MLL Classifier', cv.WINDOW_FREERATIO)
    cv.imshow('MLL Classifier', img_mll)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # POST-PROCESSOR ##########################

    # ccs_text, ccs_non_text, img_text = Postprocessor(img_text, ccs_text, ccs_non_text).postprocess()
    #
    # post = resized_img.copy()
    # cv.drawContours(post, [cc.get_contour()
    #                        for cc in ccs_text], -1, (0, 255, 0), 2)
    # cv.drawContours(post, [cc.get_contour()
    #                        for cc in ccs_non_text], -1, (0, 0, 255), 2)
    #
    # cv.namedWindow('Post-processed', cv.WINDOW_FREERATIO)
    # cv.imshow('Post-processed', post)
    # if cv.waitKey(0) & 0xff == 27:
    #     cv.destroyAllWindows()

    # TEXT SEGMENTATION ##########################
    ccs_text, ccs_non_text = TextSegmenter(img_text, ccs_text, ccs_non_text, resized_img).segment_text()

    segmented = resized_img.copy()

    cv.drawContours(segmented, [cc.get_contour()
                                for cc in ccs_text], -1, (0, 255, 0), 2)
    cv.drawContours(segmented, [cc.get_contour()
                                for cc in ccs_non_text], -1, (0, 0, 255), 2)
    cv.namedWindow('Segmented', cv.WINDOW_FREERATIO)
    cv.imshow('Segmented', segmented)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

    # Non-text classification ##########################
    ccs = NonTextClassifier(resized_img.shape[:2], ccs_text, ccs_non_text).classify_non_text_elements()

    # Region refinement and labeling ##########################
    region_refiner = RegionRefiner()
    labeled = region_refiner.label_regions(resized_img, ccs)

    cv.namedWindow('Labeled', cv.WINDOW_FREERATIO)
    cv.imshow('Labeled', labeled)
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
    # main('img/la.png')
    main('img/PRImA Layout Analysis Dataset/Images/00000880.tif')
