import cv2 as cv

T_AREA = 6
T_INSIDE = 4


def main():
    src = cv.imread('img/Bebe.png', cv.IMREAD_UNCHANGED)

    thresh = threshold(src)

    ccs = find_ccs(thresh)

    ccs_text, ccs_non_text = h_filter(ccs)

    con_img = cv.drawContours(src, ccs_text, -1, (0, 255, 0), 3)

    con_img = cv.drawContours(con_img, ccs_non_text, -1, (255, 0, 0), 3)

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
    return ccs


def h_filter(ccs):
    """
    Heuristic filter.

    Filters out not-text connected components (CCs) from the input ccs.
    
    Parameters:
        css: CCs (contours) of an image

    Returns:
        ccs_text: CCs of text components
        ccs_img: CCs of non_text components
    """

    ccs_text = []
    ccs_non_text = []
    for i, cc in enumerate(ccs):
        area = cv.contourArea(cc)
        if area < T_AREA:
            ccs_non_text.append(ccs[i])
            continue
        else:
            ccs_text.append(ccs[i])

    # TODO: Implement inc(i) > T_INSIDE belongs to ccs_non_text

    return ccs_text, ccs_non_text


def display_img(title, img):
    cv.imshow(title, img)

    # De-allocate any associated memory usage
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
