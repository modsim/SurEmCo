# by christian c. sachs

from PySide.QtCore import *
from PySide.QtGui import *

from yaval import Visualizer, VispyPlugin, MatplotlibPlugin, Values

def read_magic_number(filename):
    with open(filename, 'rb') as fp:
        return fp.read(4)

def is_png(magic_number):
    return magic_number == b'\x89PNG'

def is_gif(magic_number):
    return magic_number == b'GIF8'  # GIF89a GIF87a

def is_jpeg(magic_number):
    return magic_number[:3] == b'\xff\xd8\xff'

def is_tiff(magic_number):
    return (
        (magic_number == b'II\x2a\x00') or
        (magic_number == b'MM\x00\x2a')
    )


def is_image_file(magic_number=None, filename=None):
    if filename is not None:
        magic_number = read_magic_number(filename)
    return is_png(magic_number) or is_gif(magic_number) or is_jpeg(magic_number) or is_tiff(magic_number)


import sys

from .misc import to_rgb8, binarization_to_contours, binarize_image, contour_to_cell, get_subset_and_snippet, markerize_identical_emitters

import numpy
import pandas
import cv2




from vispy.scene import SceneCanvas, visuals
from vispy.visuals import ImageVisual, transforms
from vispy import app

def load_dataset(filename):
    return pandas.read_table(filename, skiprows=0, header=1, sep=' ')

def prepare_dataset(data):
    lookup = tuple(data.columns)

    mapping_table = {
        #new!
        ('#amplitude(photoelectrons),', 'x0(pixels),', 'y0(pixels),', 'simga_x(pixels),', 'sigma_y(pixels),', 'background(photoelectrons),', 'z_position(pixels),', 'quality,', 'CNR,', 'localization_precision_x(pixels),', 'localization_precision_y(pixels),', 'localization_precision_z(pixels),', 'correlation_coefficient,', 'frame'):
        {
        '#amplitude(photoelectrons),': 'amp',
        'x0(pixels),': 'x',
        'y0(pixels),': 'y',
        'simga_x(pixels),': 'sigma_x',
        'sigma_y(pixels),': 'sigma_y',
        'background(photoelectrons),': 'back',
        'z_position(pixels),': 'z',
        'quality,': 'quality',
        'CNR,': 'cnr',
        'localization_precision_x(pixels),': 'locprec_x',
        'localization_precision_y(pixels),': 'locprec_y',
        'localization_precision_z(pixels),': 'locprec_z',
        'correlation_coefficient,': 'corr_coeff',
        'frame': 'frame'
        },
        #old
        ('#amplitude(photonelectrons),', 'x0(pixels),', 'y0(pixels),', 'simga_x(pixels),', 'sigma_y(pixels),', 'backgroud(photonelectrons),', 'z_position(pixels),', 'quality,', 'CNR,', 'localiztion_precision_x(pixels),', 'localiztion_precision_y(pixels),', 'localiztion_precision_z(pixels),', 'correlation_coefficient,', 'frame'):
        {
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
        }
    }

    try:
        mapping = mapping_table[lookup]
    except IndexError:
        print("The super resolution tabular format is stored in an unsupported way.")
        raise
    data.rename(columns=mapping, inplace=True)

    if not (data.locprec_x == data.locprec_y).all():
        print("Warning! Localization precision is not x/y identical!")
        data['locprec'] = (data.locprec_x + data.locprec_y) / 2.0
    else:
        data['locprec'] = data.locprec_x

    if not (data.sigma_x == data.sigma_y).all():
        print("Warning! Sigma is not x/y identical!")
        data['sigma'] = (data.sigma_x + data.sigma_x) / 2.0
    else:
        data['sigma'] = data.sigma_x

    return data


class SuperresolutionTracking(Visualizer):

    title = "Superresolution Cell Detection & Emitter Tracking - by Christian C. Sachs, ModSim Group, IBG-1, FZ-Jülich - Preliminary software, do not rely on results!"

    def visualization(self):
        disable_detection = False

        if '--disable-detection' in sys.argv:
            disable_detection = True
            del sys.argv[sys.argv.index('--disable-detection')]


        if len(sys.argv) > 2:

            filenames = sys.argv[1:]

        else:

            filenames, _ = QFileDialog().getOpenFileNames()

            if len(filenames) == 1:
                while True:
                    new_fnames, _ = QFileDialog().getOpenFileNames()
                    print(new_fnames)
                    if len(new_fnames) == 0:
                        break
                    else:
                        filenames += new_fnames

        average_file, tabular_files = None, []

        for filename in filenames:
            if is_image_file(filename=filename):
                average_file = filename
            else:
                tabular_files.append(filename)

        if average_file is None or tabular_files is []:
            sys.exit(1)

        image = cv2.imread(average_file, -1)

        if len(image.shape) == 3:
            image = image.mean(axis=2)




        datasets = []

        for tabular_file in sorted(tabular_files):
            print("Reading %s" % (tabular_file,))

            local_data = load_dataset(tabular_file)
            local_data = prepare_dataset(local_data)

            datasets.append(local_data)

        data = datasets[0]
        for local_data in datasets[1:]:
            maximum_frame = data.frame.max()

            local_data.frame += maximum_frame + 1  # frame starts at 0 so we add 1
            data = data.append(local_data)

        print("Last frame is %d" % (data.frame.max(),))

        canvas = to_rgb8(image)

        def _dummy_cell():
            from .misc import NeatDict
            cell = NeatDict()
            cell.contour = []
            cell.hull = []

            cell.ellipse = []
            cell.bb = []

            cell.subset = data
            return cell

        if disable_detection:
            cells = [_dummy_cell()]
        else:

            binarization = binarize_image(image)
            cells = [contour_to_cell(contour) for contour in binarization_to_contours(binarization)]

            for cell in cells:
                get_subset_and_snippet(cell, data, image)
            #    markerize_identical_emitters(cell)

            for cell in cells:
                cv2.drawContours(canvas, [cell['hull']], -1, (0, 255, 0))
            for n, cell in enumerate(cells):
                cv2.putText(canvas, "%d" % (n,), tuple(map(int, cv2.minAreaRect(cell['hull'])[0])), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0))

            if len(cells) == 0:
                print("WARNING: Cell detection was requested, but no cells were detected! Falling back to whole image as ROI")

                cells = [_dummy_cell()]

        #####

        plugin = VispyPlugin()
        self.register_plugin(plugin)
        plugin.add_pan_zoom_camera()
        rendered_image = plugin.add_image(canvas)

        precision_label = Values.Label('precision', "Precision: %.4f µm", 0.0)
        sigma_label = Values.Label('sigma', "Sigma: %.4f µm", 0.0)


        for v in [
            precision_label,
            sigma_label,
            Values.ListValue(None, [0, 1], 'live'),
            Values.ListValue(None, list(range(len(cells))), 'cell'),
            Values.ListValue(None, [
                'custom_moving_brute', 'custom_moving_kd', 'custom_static_brute_locprec', 'custom_static_brute_sigma', 'custom_static_kd_locprec', 'custom_static_kd_sigma', 'trackpy'
            ], 'tracker'),
            Values.IntValue('exposure_time', 86, minimum=0, maximum=1000, unit="ms"),
            Values.IntValue('calibration', 65, minimum=0, maximum=1000, unit="nm·pixel⁻¹"),
            Values.FloatValue('maximum_displacement', 0.195, minimum=0, maximum=20, unit="µm"),
            Values.IntValue('maximum_blink_dark', 1, minimum=0, maximum=100, unit="frame(s)"),
            Values.Action('refresh'),
            Values.Action('show_all'),
            Values.Action('analyse_all'),
            Values.Action('msd'),
            Values.Action('msd_all'),
            Values.Action('clear'),
            Values.Action('quit')
        ]:
            self.add_value(v)


        scatter = visuals.Markers()
        lines = visuals.Line()

        plugin.view.add(scatter)
        plugin.view.add(lines)


        result_table = [{} for n in range(len(cells))]




        def _empty():

            for cell in cells:
                cell['render_data'] = None
                cell['render_conn'] = None

                cell['tracked'] = None
                cell['emsd'] = None

            for n, result in enumerate(result_table):
                result.update({
                    "Cell #": str(n),
                    "Max Displacement (µm)": float('nan'),
                    "Max Dark (frames)": float('nan'),
                    "D": float('nan'),
                    "D_MSD": float('nan'),
                    #"D_LAGT": float('nan'),
                    "Count": float('nan'),
                    "Mean Loc Precision (µm)": float('nan'),
                    "Mean Loc Sigma (µm)": float('nan'),
                    "Ellipse small (µm)": float('nan'),
                    "Ellipse long (µm)": float('nan'),
                    "Convex hull area (µm²)": float('nan'),
                    "Count / µm²": float('nan'),
                    "nm per pixel": 0,
                    "exposure (ms)": 0,
                    "Filename AVG": '',
                    "Filenames Results": '',
                    "Filename AVG full": '',
                    "Filenames Results full": ''
                })

            self.output_model.update_data(result_table)

        _empty()

        precision_in_pixel = data.locprec.mean()
        sigma_in_pixel = data.sigma.mean()

        def _update(values):
            MICRON_PER_PIXEL = (values.calibration / 1000.0)
            FRAMES_PER_SECOND = 1000.0/values.exposure_time

            precision_label.update(precision_in_pixel * MICRON_PER_PIXEL)
            sigma_label.update(sigma_in_pixel * MICRON_PER_PIXEL)

            def calc_diff(n):

                import trackpy
                cell = cells[n]

                tracked = cell['tracked']


                # had some discussions with slavko
                if False:
                    from matplotlib import pyplot

                    im = trackpy.imsd(tracked, MICRON_PER_PIXEL, FRAMES_PER_SECOND)
                    fig, ax = pyplot.subplots()
                    ax.plot(im.index, im, 'k-', alpha=0.01)
                    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
                           xlabel='lag time $t$')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    fig.savefig('_test.pdf')

                #for pid, trajectory in tracked.groupby('particle'):
                #    msd = trackpy.msd(trajectory, MICRON_PER_PIXEL, FRAMES_PER_SECOND)
                #    pylab.plot(numpy.array(msd.index), numpy.array(msd))



                #imsd = trackpy.imsd(tracked, MICRON_PER_PIXEL, FRAMES_PER_SECOND)
                #print(imsd)
                emsd = trackpy.emsd(tracked, MICRON_PER_PIXEL, FRAMES_PER_SECOND)
                cell['emsd'] = emsd
                lagt = numpy.array(emsd.index)
                msd = numpy.array(emsd)


                dimensionality = 2  # we can observe only two dimensions, then it should be two?

                D = numpy.array(msd / (lagt * 2 * dimensionality))

                result_table[n]["D"] = D.mean()#[0]
                #result_table[n]["D_MSD"] = msd[0]
                #result_table[n]["D_LAGT"] = lagt[0]

                #print("Cell", n)
                #print(numpy.c_[lagt, msd, D])

                self.output_model.update_data(result_table)



            def redo(n):

                cell = cells[n]
                subset = cell['subset']

                if values.tracker == 'trackpy':
                    import trackpy
                    if 'trackpy_tracked' in cell and 'last_memory' in cell and cell['last_memory'] == values.maximum_blink_dark and \
                        'last_displacement' in cell and cell['last_displacement'] == values.maximum_displacement:
                        tracked = cell['trackpy_tracked']
                    else:
                        tracked = trackpy.link_df(subset, values.maximum_displacement / MICRON_PER_PIXEL, memory=values.maximum_blink_dark, link_strategy='nonrecursive')
                        cell['last_memory'] = values.maximum_blink_dark
                        cell['last_displacement'] = values.maximum_displacement
                        cell['trackpy_tracked'] = tracked
                elif values.tracker.startswith('custom'):

                    from .tracker import Tracker

                    tracker = Tracker()

                    transfer = tracker.empty_track_input_type(len(subset))

                    transfer['x'] = subset.x
                    transfer['y'] = subset.y
                    transfer['frame'] = subset.frame
                    transfer['index'] = range(len(transfer))


                    # localization precision is always the same
                    # sigma is different

                    strategy, mode, what = {
                        'custom_moving_brute': (tracker.STRATEGY_BRUTE_FORCE, tracker.TRACKING_MOVING, 'locprec'),
                        'custom_static_brute_locprec': (tracker.STRATEGY_BRUTE_FORCE, tracker.TRACKING_STATIC, 'locprec'),
                        'custom_static_brute_sigma': (tracker.STRATEGY_BRUTE_FORCE, tracker.TRACKING_STATIC, 'sigma'),
                        'custom_moving_kd': (tracker.STRATEGY_KDTREE, tracker.TRACKING_MOVING, 'locprec'),
                        'custom_static_kd_locprec': (tracker.STRATEGY_KDTREE, tracker.TRACKING_STATIC, 'locprec'),
                        'custom_static_kd_sigma': (tracker.STRATEGY_KDTREE, tracker.TRACKING_STATIC, 'sigma')
                    }[values.tracker]

                    if what == 'locprec':
                        transfer['precision'] = subset.locprec
                    else:
                        transfer['precision'] = subset.sigma

                    tracker.track(transfer, values.maximum_displacement / MICRON_PER_PIXEL, values.maximum_blink_dark, mode, strategy)


                    #numpy.save("_tmp.npy", transfer)




                    def my_emsd(data):
                        maxframe = 0
                        tracks = []
                        for label in sorted(numpy.unique(data['label'])):
                            trace = data[data['label']==label].copy()
                            if len(trace) == 1:
                                continue
                            trace = numpy.sort(trace, order='frame')
                            relframe = trace['frame'].copy()
                            relframe -= relframe.min()
                            relframemax = relframe.max()
                            if relframemax > maxframe:
                                maxframe = relframemax
                            tracks.append((trace, relframe))

                        result = numpy.ones((len(tracks), maxframe+1)) * float('nan')

                        for n, (trace, relframe) in enumerate(tracks):
                            for m, sqd in zip(relframe, trace['square_displacement']):
                                result[n, m] = sqd

                        y = numpy.nanmean(result, axis=0)

                        x = numpy.linspace(0, len(y)-1, len(y))

                        Q=numpy.c_[x/FRAMES_PER_SECOND, (y * (MICRON_PER_PIXEL**2))][1:, :]

                        lagt = Q[:, 0]
                        msd = Q[:, 1]


                        dimensionality = 2  # we can observe only two dimensions, then it should be two?

                        D = numpy.array(msd / (lagt * 2 * dimensionality))
                        return D, Q


                    #D, Q = my_emsd(transfer)
                    result_table[n]["D"] = tracker.msd(transfer, MICRON_PER_PIXEL, FRAMES_PER_SECOND)  #D.mean()
                    #result_table[n]["D_MSD"] = float('nan')
                    #result_table[n]["D_MSD"] = (Q[:, 1]/Q[:, 0]).mean()

                    tracked = subset.copy()
                    tracked['particle'] = transfer['label']
                    if hasattr(tracked, 'sort_values'):
                        tracked = tracked.sort_values(by=['particle', 'frame']) # check this, it does not work with older pandas!
                    else:
                        tracked = tracked.sort(columns=['particle', 'frame'])





                cell['tracked'] = tracked

                subset = tracked
                conn = numpy.array(tracked.particle, dtype=numpy.uint32)

                nconn = numpy.zeros(len(conn), dtype=numpy.bool)

                nconn[:-1] = conn[:-1] == conn[1:]

                conn = nconn

                #subset = data

                render_data = numpy.c_[subset.x, subset.y, subset.frame]

                cell['render_data'] = render_data
                cell['render_conn'] = conn

                scatter.set_data(render_data, edge_color=None, face_color=(1, 1, 1, 0.5), size=5)
                lines.set_data(render_data, color=(1, 1, 1, 0.5), connect=conn)

                scatter.update()
                lines.update()

                plugin.view.camera = 'turntable'

                result_table[n]["Max Displacement (µm)"] = values.maximum_displacement
                result_table[n]["Max Dark (frames)"] = values.maximum_blink_dark

                import os.path


                result_table[n]["Filename AVG"] = os.path.basename(average_file)
                result_table[n]["Filenames Results"] = "::".join(os.path.basename(t) for t in tabular_files)
                result_table[n]["Filename AVG full"] = average_file
                result_table[n]["Filenames Results full"] = "::".join(tabular_files)

                #print(tracked)
                the_count = int(tracked.particle.max())
                result_table[n]["Count"] = the_count

                result_table[n]["Mean Loc Precision (µm)"] = subset.locprec.mean() * MICRON_PER_PIXEL
                result_table[n]["Mean Loc Sigma (µm)"] = subset.sigma.mean() * MICRON_PER_PIXEL

                smaller_axis = min(*cell.ellipse[1])
                larger_axis = max(*cell.ellipse[1])

                result_table[n]["Ellipse small (µm)"] = smaller_axis * MICRON_PER_PIXEL
                result_table[n]["Ellipse long (µm)"] = larger_axis * MICRON_PER_PIXEL

                if len(cell.contour) > 0:
                    the_area = cv2.contourArea(cell.contour)
                    the_area *= MICRON_PER_PIXEL**2
                    result_table[n]["Convex hull area (µm²)"] = the_area
                    result_table[n]["Count / µm²"] = the_count / the_area

                self.output_model.update_data(result_table)


            if values['clear']:
                _empty()


            if values.quit:
                sys.exit(1)

            if values.refresh:
                redo(values.cell)

            if values.msd:
                calc_diff(values.cell)

            if values.msd_all:
                for n in range(len(cells)):
                    calc_diff(n)
                print("Done.")

            if values.analyse_all or (values.tracker.startswith('custom') and values.live == 1):
                for n in range(len(cells)):
                    redo(n)
                print("Done.")

                if False:
                    import threading
                    threads = [threading.Thread(target=redo, args=(n,)) for n in range(len(cells))]
                    print("*")
                    for t in threads:
                        t.start()
                        print("*")
                    for t in threads:
                        t.join()
                        print("*")

                    print("Done.")

                values.show_all = True

            if values.show_all:

                try:

                    render_data = numpy.concatenate([cell['render_data'] for cell in cells if cell['render_data'] is not None])
                    render_conn = numpy.concatenate([cell['render_conn'] for cell in cells if cell['render_conn'] is not None])

                    scatter.set_data(render_data, edge_color=None, face_color=(1, 1, 1, 0.5), size=5)
                    lines.set_data(render_data, color=(1, 1, 1, 0.5), connect=render_conn)

                    scatter.update()
                    lines.update()
                except ValueError:
                    pass
        return _update

#####
"""
512 x 512 ... 0.065 um per pixel
2560 x 2560 ... 0.013 um per pixel


Average Ribosome diffusion rate: 0.04 +- 0.01 square micrometer per second
assumption: all fluorescent ribosomal proteins are bound in ribosomes

exposure/timing:
50ms exposure + 36ms delay
(exposure every 86ms)

"""
#####

def main():

    SuperresolutionTracking.run()

    pass

if __name__ == '__main__':
    main()