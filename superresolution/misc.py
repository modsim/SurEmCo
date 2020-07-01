import re
from itertools import chain

import numpy as np
import numexpr
import cv2

from scipy.spatial import cKDTree as KDTree


# from matplotlib import pyplot
# pyplot.rcParams['figure.figsize'] = (40, 8)
# pyplot.rcParams['figure.dpi'] = 150
# pyplot.rcParams['image.cmap'] = 'gray'

def num_tokenize(file_name):
    def try_int(fragment):
        try:
            fragment_int = int(fragment)
            if str(fragment_int) == fragment:
                return fragment_int
        except ValueError:
            pass
        return fragment

    return tuple(try_int(fragment) for fragment in re.split('(\d+)', file_name))


# ## GENERIC HELPERS
class NeatDict(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]


def _cv2_get_integral_image_and_squared(image):
    ints, intss = cv2.integral2(image, sdepth=cv2.CV_64F)
    return ints[1:, 1:], intss[1:, 1:]


get_integral_image_and_squared = _cv2_get_integral_image_and_squared


def means_and_stddev(image, wr=15):
    enlarged = np.zeros((image.shape[0] + 2 * wr, image.shape[1] + 2 * wr), np.double)

    enlarged[wr:-wr, wr:-wr] = image
    enlarged[0:wr] = enlarged[wr + 1, :]
    enlarged[-wr:] = enlarged[-wr - 1, :]
    for n in range(wr):
        enlarged[:, n] = enlarged[:, wr]
        enlarged[:, -n] = enlarged[:, -wr - 1]

    ints, intss = get_integral_image_and_squared(enlarged)

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


def sauvola(image, wr=15, k=0.5, r=128):
    means, stddev = means_and_stddev(image, wr)
    return numexpr.evaluate("image > (means * (1.0 + k * ((stddev / r) - 1.0)))")


def blur_gaussian(image, sigma=1.0):
    return cv2.GaussianBlur(image, ksize=(-1, -1), sigmaX=sigma)


def blur_box(image, width_x=1, width_y=None):
    if width_y is None:
        width_y = width_x
    return cv2.blur(image, (width_x, width_y))


def rotate_image(image, angle):
    return cv2.warpAffine(image,
                          cv2.getRotationMatrix2D((image.shape[1] * 0.5, image.shape[0] * 0.5), angle, 1.0),
                          (image.shape[1], image.shape[0]))


#####################


def binarize_image(image):
    image = image.astype(float)
    image /= blur_gaussian(image, 50)

    image -= image.min()
    image /= image.max()
    image = 1 - image
    image *= 255

    # pyplot.imshow(image)
    binarization = sauvola(image, wr=15, k=0.1, r=140)

    return binarization


def binarization_to_contours(binarization, minimum_area=100, maximum_area=10000):
    contours, hierarchy = cv2.findContours(binarization.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [contour for contour in contours if minimum_area < cv2.contourArea(contour) < maximum_area]


def contour_to_cell(contour):
    cell = NeatDict()
    cell.contour = contour
    cell.hull = cv2.convexHull(contour)

    cell.ellipse = cv2.fitEllipse(cell.hull)
    cell.bb = cv2.boundingRect(cell.hull)
    return cell


def get_subset_and_snippet(cell, data, image, border=0.0):
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

    subset = data.query('@x < x < (@x+@w) and @y < y < (@y+@h)')

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

    mask = [cv2.pointPolygonTest(hull_to_use.astype(np.int32), (point_x, point_y),
                                 measureDist=False) >= 0
            for point_x, point_y in zip(subset.x, subset.y)]

    subset = subset[mask]

    cell.subset = subset

    # return
    #
    # angle = cell.ellipse[2]
    #
    # subset['new_x'] = subset.x - x
    # subset['new_y'] = subset.y - y
    #
    # cell.snippet = image[int(y):int(y + h), int(x):int(x + w)]
    #
    # corr_angle = np.deg2rad(-angle + 90)
    # mid_x, mid_y = (0.5 * w), (0.5 * h)
    # subset['rot_x'] = (subset.new_x - mid_x) * np.cos(corr_angle) - (subset.new_y - mid_y) * np.sin(
    #     corr_angle) + mid_x
    # subset['rot_y'] = (subset.new_y - mid_y) * np.cos(corr_angle) + (subset.new_x - mid_x) * np.sin(
    #     corr_angle) + mid_y
    #
    # cell.rot_snippet = rotate_image(cell.snippet, np.rad2deg(-corr_angle))
    #
    # subset['abs_rot_x'] = subset.rot_x - subset.rot_x.min()
    # subset['abs_rot_y'] = subset.rot_y - subset.rot_y.min()


def markerize_identical_emitters(cell):
    # sigma_x, sigma_y = constant

    # subset.sort_values(by='x')

    xy, precision, frame = np.c_[np.array(cell.subset.x), np.array(cell.subset.y)], np.array(
        cell.subset.locprec_x), np.array(cell.subset.frame)

    marker = [-1] * len(xy)

    tree = KDTree(xy)

    for pnum, point in enumerate(xy):
        if marker[pnum] != -1:
            continue

        marker[pnum] = pnum

        delta = precision[pnum] + np.finfo(float).eps
        result = tree.query_ball_point(point, delta)
        for idx in result:
            if marker[idx] != -1:
                # print("collission")
                # continue
                pass
            marker[idx] = pnum

    cell.count = len(np.unique(marker))
    cell.marker = marker

    new = [False] * len(xy)
    memo = set()
    for pos, m in enumerate(marker):
        if m not in memo:
            new[pos] = True
            memo.add(m)
    cell.sub_subset = cell.subset[new]

    # print(len(marker))
    # print(len(np.unique(marker)))


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

        self.tracked = None
        self.emsd = None


def contour_to_mesh(contour, frame_min, frame_max):
    if len(contour) == 0:
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    mesh_data = []

    for point_a, point_b in chain(
            zip(contour[:, 0, :], contour[1:, 0, :]),
            [(contour[-1, 0, :], contour[0, 0, :])]
    ):
        mesh_data.append([point_a[0], point_a[1], frame_min])
        mesh_data.append([point_a[0], point_a[1], frame_max])
        mesh_data.append([point_b[0], point_b[1], frame_min])

        mesh_data.append([point_b[0], point_b[1], frame_min])
        mesh_data.append([point_b[0], point_b[1], frame_max])
        mesh_data.append([point_a[0], point_a[1], frame_max])

    mesh_data = np.array(mesh_data)

    return mesh_data
