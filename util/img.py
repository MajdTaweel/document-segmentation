import cv2 as cv


def show_and_wait(title, img):
    cv.namedWindow(title, cv.WINDOW_KEEPRATIO)
    cv.imshow(title, img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


def draw_contours_then_show_and_wait(title, img, ccs_and_colors_list):
    img = img.copy()
    for ccs_and_color in ccs_and_colors_list:
        cv.drawContours(img, [cc.get_contour() for cc in ccs_and_color[0]], -1, ccs_and_color[1], 2)
    show_and_wait(title, img)


def read_img(path):
    return cv.imread(path, cv.IMREAD_UNCHANGED)


def write_img(path, img):
    cv.imwrite(path, img)
