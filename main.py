import sys
import cv2 as cv
from binarizer import Binarizer
from heuristic_filter import HeuristicFilter


def main(path):
    src = cv.imread(path, cv.IMREAD_UNCHANGED)

    binarizer = Binarizer(src)

    img_bin = binarizer.binarize()

    h_filter = HeuristicFilter(img_bin)

    ccs_text, ccs_non_text = h_filter.filter()

    con_img = cv.drawContours(src, ccs_text, -1, (0, 255, 0), 2)

    con_img = cv.drawContours(con_img, ccs_non_text, -1, (0, 0, 255), 2)

    cv.imshow('Contours', con_img)

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
    args = sys.argv[1:]
    if len(args) == 0:
        print('Argument missing. Taking argument from input:')
        args.append(input())
    elif len(args) > 1:
        raise Exception(f'Too many arguments: {len(args)}. Only one argument is required.')

    main(args[0])
