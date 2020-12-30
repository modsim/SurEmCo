# SurEmCo - Miscellaneous functions

import re
from itertools import chain

import numpy as np
import numexpr
import cv2


def num_tokenize(file_name):
    def try_int(fragment):
        try:
            fragment_int = int(fragment)
            if str(fragment_int) == fragment:
                return fragment_int
        except ValueError:
            pass
        return fragment

    return tuple(try_int(fragment) for fragment in re.split(r'(\d+)', file_name))


# noinspection PyUnusedLocal
def means_and_stddev(image, wr=15):
    enlarged = np.zeros((image.shape[0] + 2 * wr, image.shape[1] + 2 * wr), np.double)

    enlarged[wr:-wr, wr:-wr] = image
    enlarged[0:wr] = enlarged[wr + 1, :]
    enlarged[-wr:] = enlarged[-wr - 1, :]
    for n in range(wr):
        enlarged[:, n] = enlarged[:, wr]
        enlarged[:, -n] = enlarged[:, -wr - 1]

    ints, intss = cv2.integral2(enlarged, sdepth=cv2.CV_64F)
    ints, intss = ints[1:, 1:], intss[1:, 1:]

    # noinspection PyPep8Naming,PyUnusedLocal
    def calc_sums(mat):
        A = mat[:-2 * wr, :-2 * wr]
        B = mat[2 * wr:, 2 * wr:]
        C = mat[:-2 * wr, 2 * wr:]
        D = mat[2 * wr:, :-2 * wr]
        return numexpr.evaluate("(A + B) - (C + D)").astype(np.float32)

    sums = calc_sums(ints)
    sumss = calc_sums(intss)

    area = (2.0 * wr + 1) ** 2

    means = sums / area

    # stddev = np.sqrt(sumss / area - means ** 2)

    stddev = numexpr.evaluate("sqrt(sumss / area - means ** 2)")

    return means, stddev


# noinspection PyUnusedLocal
def sauvola(image, wr=15, k=0.5, r=128.0):
    means, stddev = means_and_stddev(image, wr)
    return numexpr.evaluate("image > (means * (1.0 + k * ((stddev / r) - 1.0)))")


def binarize_image(image):
    image = image.astype(float)
    image /= cv2.GaussianBlur(image, ksize=(-1, -1), sigmaX=50.0)

    image -= image.min()
    image /= image.max()
    image = 1 - image
    image *= 255

    # pyplot.imshow(image)
    binarization = sauvola(image, wr=15, k=0.1, r=140.0)

    return binarization


def binarization_to_contours(binarization, minimum_area=100.0, maximum_area=10000.0):
    contours, hierarchy = cv2.findContours(binarization.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [contour for contour in contours if minimum_area < cv2.contourArea(contour) < maximum_area]


def get_subset_and_snippet(cell, data, border=0.0):
    x, y, w, h = cell.bb

    longest_edge = round(np.sqrt(w ** 2.0 + h ** 2.0))
    w_d = longest_edge
    h_d = longest_edge
    x -= w_d // 2
    w += w_d

    y -= h_d // 2
    h += h_d

    x, y = max(0, x), max(0, y)

    x -= border
    y -= border
    w += 2 * border
    h += 2 * border

    subset = data.query('@x <= x <= (@x+@w) and @y <= y <= (@y+@h)')

    hull_to_use = cell.hull

    if border > 0.0:
        hull_to_use = hull_to_use.astype(np.float32)

        mi0, mi1 = hull_to_use[:, 0, 0].min(), hull_to_use[:, 0, 1].min()
        hull_to_use[:, 0, 0] -= mi0
        hull_to_use[:, 0, 1] -= mi1

        ma0, ma1 = hull_to_use[:, 0, 0].max(), hull_to_use[:, 0, 1].max()

        hull_to_use[:, 0, 0] -= ma0/2
        hull_to_use[:, 0, 1] -= ma1/2

        hull_to_use[:, 0, 0] *= (ma0 + border) / ma0
        hull_to_use[:, 0, 1] *= (ma1 + border) / ma1

        hull_to_use[:, 0, 0] += ma0 / 2
        hull_to_use[:, 0, 1] += ma1 / 2

        hull_to_use[:, 0, 0] += mi0
        hull_to_use[:, 0, 1] += mi1

    cell.render_hull = cell.hull[:, 0, :]
    cell.render_hull_bordered = hull_to_use[:, 0, :]

    mask = [cv2.pointPolygonTest(hull_to_use.astype(np.int32), (point_x, point_y),
                                 measureDist=False) >= 0 for point_x, point_y in zip(subset.x, subset.y)]

    subset = subset[mask]

    cell.subset = subset


def to_rgb8(image):
    new_mix = np.zeros(image.shape + (3,), dtype=np.uint8)
    incoming = image.astype(float)
    incoming -= incoming.min()
    incoming /= incoming.max()
    incoming *= 255
    new_mix[:, :, 2] = new_mix[:, :, 1] = new_mix[:, :, 0] = incoming.astype(np.uint8)
    return new_mix


class Cell:
    def __init__(self, subset=None, contour=None, name=''):
        if contour is None:
            contour = []

            self.contour = contour
            self.hull = np.array([])
            self.ellipse = [0.0, 0.0]
            self.bb = [0.0, 0.0, 0.0, 0.0]
        else:
            self.contour = contour
            self.hull = cv2.convexHull(contour)

            self.ellipse = cv2.fitEllipse(self.hull)
            self.bb = cv2.boundingRect(self.hull)

        self.subset = subset

        self.name = name

        self.render_data = None
        self.render_conn = None
        self.render_mesh = None

        self.render_hull = None
        self.render_hull_bordered = None

        self.tracked = None


def contour_to_mesh(contour, frame_min, frame_max, interval=250):
    if len(contour) == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    mesh_data = []

    intervals = np.arange(frame_min, frame_max, interval)
    if intervals[-1] != frame_max:
        intervals = np.r_[intervals, frame_max]

    for low, high in zip(intervals, intervals[1:]):
        for point_a, point_b in chain(
                zip(contour[:, 0, :], contour[1:, 0, :]),
                [(contour[-1, 0, :], contour[0, 0, :])]
        ):
            mesh_data.append([point_a[0], point_a[1], low])
            mesh_data.append([point_a[0], point_a[1], high])
            mesh_data.append([point_b[0], point_b[1], low])

            mesh_data.append([point_b[0], point_b[1], low])
            mesh_data.append([point_b[0], point_b[1], high])
            mesh_data.append([point_a[0], point_a[1], high])

    mesh_data = np.array(mesh_data)

    return mesh_data
