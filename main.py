import cv2 as cv
import numpy as np

T_AREA = 6
T_INSIDE = 4


def main():
    src = cv.imread('img/triangles.png', cv.IMREAD_UNCHANGED)

    thresh = threshold(src)

    ccs, hierarchy = find_ccs(thresh)

    ccs_text, ccs_non_text = h_filter(thresh, ccs, hierarchy)

    con_img = cv.drawContours(src, ccs_text, -1, (0, 255, 0), 2)

    con_img = cv.drawContours(con_img, ccs_non_text, -1, (0, 0, 255), 2)

    display_img('Contours', con_img)


def threshold(img):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    thresh = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 199, 5)
    return thresh


def find_ccs(img):
    ccs, hierarchy = cv.findContours(
        img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return ccs, hierarchy


def h_filter(img, ccs, hierarchy):
    """
    Heuristic filter.

    Filters out not-text connected components (CCs) from the input ccs.

    Parameters:
        css: CCs (contours) of an image

    Returns:
        ccs_text: CCs of text components
        ccs_img: CCs of non_text components
    """

    ccs_noise = []
    for i, cc in enumerate(ccs):
        area = cv.contourArea(cc)
        if area < T_AREA:
            ccs_noise.append(cc)
            continue

    img_denoise, ccs_denoise, hierarchy_denoise = filter_then_get_ccs_and_hierarchy(
        img, ccs_noise)

    ccs_non_text = find_non_text_ccs(ccs_denoise, hierarchy_denoise)

    img_text, ccs_text, hierarchy_text = filter_then_get_ccs_and_hierarchy(
        img_denoise, ccs_non_text)

    return ccs_text, ccs_non_text


def find_non_text_ccs(ccs, hierarchy):
    ccs_non_text = []
    for i, cc in enumerate(ccs):
        if has_descendants_more_than(ccs, hierarchy, i, T_INSIDE):
            ccs_non_text.append(cc)

    return ccs_non_text


def has_descendants_more_than(ccs, hierarchy, current, t_inside):
    # hierarchy[0][i] => [Next, Previous, First_Child, Parent] of element i
    if hierarchy[0][current][3] != -1:
        return False

    num_descendants = get_num_descendants(hierarchy, hierarchy[0][current][2])
    return num_descendants > t_inside


def get_num_descendants(hierarchy, current, dir=0):
    if current == -1:
        return 0

    num_descendants = 0

    if dir != -1:
        num_descendants += get_num_descendants(hierarchy,
                                               hierarchy[0][current][0], 1)

    if dir != 1:
        num_descendants += get_num_descendants(hierarchy,
                                               hierarchy[0][current][1], -1)

    num_descendants += get_num_descendants(hierarchy, hierarchy[0][current][2])

    return num_descendants + 1


def get_bounding_rect(cc):
    x, y, w, h = cv.boundingRect(cc)
    return x, y, w, h


def draw_bounding_rect(img, x, y, w, h):
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


def filter_then_get_ccs_and_hierarchy(img, rm_ccs):
    img_filtered = cv.drawContours(img, rm_ccs, -1, (255, 255, 255), -1)
    ccs, hierarchy = find_ccs(img_filtered)
    return img_filtered, ccs, hierarchy


def display_img(title, img):
    cv.imshow(title, img)

    # De-allocate any associated memory usage
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
