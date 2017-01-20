import numpy
import numexpr
import pandas
import cv2
from scipy.spatial import cKDTree as KDTree


# from matplotlib import pyplot
# pyplot.rcParams['figure.figsize'] = (40, 8)
# pyplot.rcParams['figure.dpi'] = 150
# pyplot.rcParams['image.cmap'] = 'gray'

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
    enlarged = numpy.zeros((image.shape[0] + 2 * wr, image.shape[1] + 2 * wr), numpy.double)

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
        return numexpr.evaluate("(A + B) - (C + D)").astype(numpy.float32)

    sums = calc_sums(ints)
    sumss = calc_sums(intss)

    area = (2.0 * wr + 1) ** 2

    means = sums / area

    # stddev = numpy.sqrt(sumss / area - means ** 2)

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
    _, contours, hierarchy = cv2.findContours(binarization.astype(numpy.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [contour for contour in contours if minimum_area < cv2.contourArea(contour) < maximum_area]


def contour_to_cell(contour):
    cell = NeatDict()
    cell.contour = contour
    cell.hull = cv2.convexHull(contour)

    cell.ellipse = cv2.fitEllipse(cell.hull)
    cell.bb = cv2.boundingRect(cell.hull)
    return cell


def get_subset_and_snippet(cell, data, image):
    x, y, w, h = cell.bb

    longest_edge = round(numpy.sqrt(w ** 2.0 + h ** 2.0))
    w_d = longest_edge
    h_d = longest_edge
    x -= w_d // 2
    w += w_d

    y -= h_d // 2
    h += h_d

    x, y = max(0, x), max(0, y)

    subset = data.query('@x < x < (@x+@w) and @y < y < (@y+@h)')
    mask = [cv2.pointPolygonTest(cell.hull, (point_x, point_y), measureDist=False) >= 0 for point_x, point_y in
            zip(subset.x, subset.y)]
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
    # corr_angle = numpy.deg2rad(-angle + 90)
    # mid_x, mid_y = (0.5 * w), (0.5 * h)
    # subset['rot_x'] = (subset.new_x - mid_x) * numpy.cos(corr_angle) - (subset.new_y - mid_y) * numpy.sin(
    #     corr_angle) + mid_x
    # subset['rot_y'] = (subset.new_y - mid_y) * numpy.cos(corr_angle) + (subset.new_x - mid_x) * numpy.sin(
    #     corr_angle) + mid_y
    #
    # cell.rot_snippet = rotate_image(cell.snippet, numpy.rad2deg(-corr_angle))
    #
    # subset['abs_rot_x'] = subset.rot_x - subset.rot_x.min()
    # subset['abs_rot_y'] = subset.rot_y - subset.rot_y.min()


def markerize_identical_emitters(cell):
    # sigma_x, sigma_y = constant

    # subset.sort_values(by='x')

    xy, precision, frame = numpy.c_[numpy.array(cell.subset.x), numpy.array(cell.subset.y)], numpy.array(
        cell.subset.locprec_x), numpy.array(cell.subset.frame)

    marker = [-1] * len(xy)

    tree = KDTree(xy)

    for pnum, point in enumerate(xy):
        if marker[pnum] != -1:
            continue

        marker[pnum] = pnum

        delta = precision[pnum] + numpy.finfo(float).eps
        result = tree.query_ball_point(point, delta)
        for idx in result:
            if marker[idx] != -1:
                # print("collission")
                # continue
                pass
            marker[idx] = pnum

    cell.count = len(numpy.unique(marker))
    cell.marker = marker

    new = [False] * len(xy)
    memo = set()
    for pos, m in enumerate(marker):
        if m not in memo:
            new[pos] = True
            memo.add(m)
    cell.sub_subset = cell.subset[new]

    # print(len(marker))
    # print(len(numpy.unique(marker)))


def to_rgb8(image):
    new_mix = numpy.zeros(image.shape + (3,), dtype=numpy.uint8)
    incoming = image.astype(float)
    incoming -= incoming.min()
    incoming /= incoming.max()
    incoming *= 255
    new_mix[:, :, 2] = new_mix[:, :, 1] = new_mix[:, :, 0] = incoming.astype(numpy.uint8)
    return new_mix


def main():
    import sys

    print("PRELIMINARY SOFTWARE by Christian C. Sachs, ModSim Group, IBG-1 FZ Jülich")
    print("Do not rely on data generated ...")

    # sys.argv

    if len(sys.argv) < 3:
        print("Usage: {} <average image.png> <tabular resultfile.txt> <output.pdf>".format(sys.argv[0]))
        return

    filename = {
        'recons': sys.argv[2],
        'average': sys.argv[1],
        'output': sys.argv[3]
    }

    filename = NeatDict(filename)
    image = cv2.imread(filename.average, -1)
    data = pandas.read_table(filename.recons, skiprows=0, header=1, sep=' ')

    data.rename(columns={
        '#amplitude(photonelectrons),': 'amp',
        'x0(pixels),': 'x',
        'y0(pixels),': 'y',
        'simga_x(pixels),': 'sigma_x',
        'sigma_y(pixels),': 'sigma_y',
        'backgroud(photonelectrons),': 'back',
        'z_position(pixels),': 'z',
        'quality,': 'quality',
        'CNR,': 'cnr',
        'localiztion_precision_x(pixels),': 'locprec_x',
        'localiztion_precision_y(pixels),': 'locprec_y',
        'localiztion_precision_z(pixels),': 'locprec_z',
        'correlation_coefficient,': 'corr_coeff',
        'frame': 'frame'
    }, inplace=True)

    binarization = binarize_image(image)
    cells = [contour_to_cell(contour) for contour in binarization_to_contours(binarization)]

    for cell in cells:
        get_subset_and_snippet(cell, data, image)
        markerize_identical_emitters(cell)

    print("Analyses finished")

    write_pdf = True

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages("%s" % (filename.output,)) as pdf:

        new_contour_image = numpy.zeros_like(binarization, dtype=numpy.uint8)
        for cell in cells:
            cv2.drawContours(new_contour_image, [cv2.boxPoints(cell.ellipse).astype(numpy.int32)], -1, 255)
            x, y, w, h = cv2.boundingRect(cell.hull)
            cv2.rectangle(new_contour_image, (x, y), (x + w, y + h), 255)
            cv2.drawContours(new_contour_image, [cell.contour], -1, 255)
            cv2.drawContours(new_contour_image, [cell.hull], -1, 255)

        f, axes = pyplot.subplots(1, 4)
        axes[0].set_title('Preliminary')
        axes[1].set_title('Test Software')
        axes[2].set_title('by Christian C. Sachs')
        axes[3].set_title('ModSim Group / IBG-1 / FZ Jülich')
        axes[0].set_rasterization_zorder(-1)
        axes[0].scatter(data.x, data.y, zorder=-2)
        axes[0].imshow(image, zorder=-3)
        axes[1].imshow(image)
        axes[2].imshow(binarization)
        axes[3].imshow(new_contour_image)
        if write_pdf:
            pdf.savefig()
        else:
            pyplot.show()
        pyplot.close()

        canvas = to_rgb8(image)
        for cell in cells:
            cv2.drawContours(canvas, [cell.hull], -1, (0, 255, 0))
        for cell in cells:
            cv2.putText(canvas, ". %d" % (cell.count,), tuple(map(int, cv2.minAreaRect(cell.hull)[0])),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        pyplot.title('Filtered Emitter Counts per Cell')
        pyplot.imshow(canvas[:, :, ::-1])

        if write_pdf:
            pdf.savefig()
        else:
            pyplot.show()
        pyplot.close()

        for n, cell in enumerate(cells):
            pyplot.figure(figsize=(40, 8))
            f, axes = pyplot.subplots(1, 5)

            alpha = 0.1

            axes[0].set_title('Overview [scatter opacity=%.2f]' % alpha)
            axes[0].set_rasterization_zorder(-1)
            axes[0].scatter(cell.subset.x, cell.subset.y, alpha=alpha, zorder=-2)
            axes[0].imshow(new_contour_image, zorder=-50)

            alpha = 0.01

            axes[1].set_title('Rotated Cell Image [raw emitters=%d, scatter opacity=%.2f]' % (len(cell.subset), alpha))
            axes[1].imshow(cell.rot_snippet, zorder=-50)
            axes[1].set_rasterization_zorder(0)
            axes[1].scatter(cell.subset.rot_x, cell.subset.rot_y, alpha=alpha, zorder=-2)
            axes[1].scatter(cell.sub_subset.rot_x, cell.sub_subset.rot_y, alpha=alpha, zorder=-1, color='red')

            hist = numpy.histogram(cell.subset.abs_rot_x, bins=int(round(cell.subset.abs_rot_x.max())))
            axes[2].set_title('Histogram [Count]')
            axes[2].set_ylabel('Emitter count [#]')
            axes[2].set_xlabel('Bin position [pixel]')
            axes[2].plot(hist[1][:-1], hist[0])

            axes[3].set_title(
                'Rotated Cell Image [filtered emitters=%d, scatter opacity=%.2f]' % (len(cell.sub_subset), alpha))
            axes[3].imshow(cell.rot_snippet, zorder=-50)
            axes[3].set_rasterization_zorder(0)
            axes[3].scatter(cell.sub_subset.rot_x, cell.sub_subset.rot_y, alpha=alpha, zorder=-1, color='red')

            hist = numpy.histogram(cell.sub_subset.abs_rot_x, bins=int(round(cell.sub_subset.abs_rot_x.max())))
            axes[4].set_title('Filtered Histogram [Count]')
            axes[4].set_ylabel('Filtered Emitter count [#]')
            axes[4].set_xlabel('Bin position [pixel]')
            axes[4].plot(hist[1][:-1], hist[0])

            if write_pdf:
                pdf.savefig()
            else:
                pyplot.show()
            pyplot.close()

            print("Done %d/%d cells." % (n + 1, len(cells)))


if __name__ == '__main__':
    main()
