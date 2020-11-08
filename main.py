import cv2 as cv
import numpy as np

T_AREA = 6
T_INSIDE = 4


def main():
    src = cv.imread('img/la.png', cv.IMREAD_UNCHANGED)

    thresh = cv.bitwise_not(threshold(src))

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

    img_denoise, ccs_denoise = filter_noise(img, ccs_noise)

    img_text, ccs_text, ccs_non_text = find_non_text(img, ccs_denoise)

    return ccs_text, ccs_non_text


def filter_noise(img, rm_ccs):
    img_filtered = cv.drawContours(img, rm_ccs, -1, (255, 255, 255), -1)
    ccs, hierarchy = find_ccs(img_filtered)
    return img_filtered, ccs


def find_non_text(img, ccs):
    ccs_non_text = []
    for i, cc in enumerate(ccs):
        if has_descendants_more_than(ccs, i, T_INSIDE):
            ccs_non_text.append(cc)
            x, y, w, h = cv.boundingRect(cc)
            img = cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)

    ccs_text, hierarchy = find_ccs(img)

    return img, ccs_text, ccs_non_text


def has_descendants_more_than(ccs, i, t_inside):
    num_descendants = 0
    x_i, y_i, w_i, h_i = cv.boundingRect(ccs[i])
    for j, cc_j in enumerate(ccs, start=i + 1):
        x_j, y_j, w_j, h_j = cv.boundingRect(cc_j)
        if x_j >= x_i and y_j >= y_i and x_j + w_j <= x_i + w_i and y_j + h_j <= y_i + h_i:
            num_descendants += 1
            if num_descendants > t_inside:
                return True

    return False


# def get_bounding_rect(cc):
#     x, y, w, h = cv.boundingRect(cc)

#     Rotated rect
#     rect = cv.minAreaRect(cnt)
#     box = cv.boxPoints(rect)
#     box = np.int0(box)
#     cv.drawContours(img,[box],0,(0,0,255),2)

#     return x, y, w, h


def display_img(title, img):
    cv.imshow(title, img)

    # De-allocate any associated memory usage
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
