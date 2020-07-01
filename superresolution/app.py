# Copyright (C) 2015-2017 Christian C. Sachs, Forschungszentrum Jülich
#####
"""
512 x 512 ... 0.065 um per pixel
2560 x 2560 ... 0.013 um per pixel


Average ribosome diffusion rate: 0.04 +- 0.01 square micrometer per second
assumption: all fluorescent ribosomal proteins are bound in ribosomes

exposure/timing:
50ms exposure + 36ms delay
(exposure every 86ms)

"""


#####

from yaval import Visualizer, Values, VispyPlugin
from yaval.qt import QFileDialog

import cv2
import numpy as np
import pandas as pd
import time
import os.path

from argparse import ArgumentParser
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from scipy.stats import linregress

from .io import is_image_file, load_dataset, prepare_dataset
from .misc import Cell, to_rgb8, binarize_image, binarization_to_contours, get_subset_and_snippet, num_tokenize, \
    contour_to_mesh

try:
    from .tracker import Tracker
except ImportError:
    print("WARNING: Custom tracker not available. Was it installed correctly?")
    Tracker = None

try:
    import trackpy
except ImportError:
    print("WARNING: TrackPy not installed.")
    trackpy = None


def create_argparser():
    parser = ArgumentParser(description="Superresolution Analyser")

    parser.add_argument("files", metavar="files", type=str, nargs='*', default=[],
                        help="input files, one must be a DIA image")
    parser.add_argument("--disable-detection", dest="disable_detection", action='store_true')
    parser.add_argument("--drift-correction", dest="drift_correction", action='store_true')
    parser.add_argument("--show-unassigned", dest="show_unassigned", action='store_true')
    parser.add_argument("--add-cell-border", dest="border", type=float, default=0.0)
    parser.add_argument("--calibration", dest="calibration", type=float, default=0.065, help="µm per pixel")
    parser.add_argument("--keep-order", dest="keep_order", action='store_true')

    return parser


class SuperresolutionTracking(Visualizer):
    title = "Superresolution Cell Detection & Emitter Tracking - " + \
            "by Christian C. Sachs, ModSim Group, IBG-1, FZ-Jülich - " + \
            "Preliminary software, do not rely on results!"

    result_table = True

    def visualization(self):
        parser = create_argparser()
        args = parser.parse_args()

        if len(args.files) < 2:
            args.files, _ = QFileDialog().getOpenFileNames()

            if len(args.files) == 1:
                while True:
                    new_fnames, _ = QFileDialog().getOpenFileNames()
                    if len(new_fnames) == 0:
                        break
                    else:
                        args.files += new_fnames

        average_file, tabular_files = None, []

        for filename in args.files:
            if is_image_file(filename=filename):
                average_file = filename
            else:
                tabular_files.append(filename)

        if average_file is None or tabular_files is []:
            raise SystemExit

        image = cv2.imread(average_file, -1)

        if len(image.shape) == 3:
            image = image.mean(axis=2)

        datasets = []

        maximum_frame = 0

        for tabular_file in (tabular_files if args.keep_order else sorted(tabular_files, key=num_tokenize)):
            print("Reading %s" % (tabular_file,))

            local_data = load_dataset(tabular_file)
            local_data = prepare_dataset(local_data)

            local_data.frame += maximum_frame

            datasets.append(local_data)

            maximum_frame = local_data.frame.max() + 1

        data = pd.concat(datasets)
        data.reset_index(inplace=True)
        data['original_index'] = data.index.copy()
        maximum_frame = data.frame.max()

        print("Last frame is %d" % (maximum_frame,))

        canvas = to_rgb8(image)

        if args.disable_detection:
            cells = [Cell(subset=data)]
        else:
            binarization = binarize_image(image)
            cells = [Cell(contour=contour) for contour in binarization_to_contours(binarization)]

            # label them
            for n, cell in enumerate(cells):
                cell.name = '%d Cell' % n

            for cell in cells:
                get_subset_and_snippet(cell, data, args.border / args.calibration)

            # markerize_identical_emitters(cell)

            # for cell in cells:
            #     cv2.drawContours(canvas, [cell.hull], -1, (0, 255, 0))
            # for n, cell in enumerate(cells):
            #     cv2.putText(canvas, "%d" % (n,), tuple(map(int, cv2.minAreaRect(cell.hull)[0])),
            #                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

            if len(cells) == 0:
                print(
                    "WARNING: Cell detection was requested, but no cells were detected!" +
                    " Falling back to whole image as ROI"
                )

                cells = [Cell(subset=data, name='0 Everything')]

            if args.show_unassigned:
                mask = np.ones(len(data), dtype=bool)

                for cell in cells:
                    mask[cell.subset.original_index] = False

                remainder = data[mask]

                del mask

                cells.append(Cell(subset=remainder, name='%d Unassigned' % len(cells)))

                cells.append(Cell(subset=data, name='%d Everything' % len(cells)))

        # this does NOT work.
        if args.drift_correction:
            print("Drift correction")

            groups_groups = [cell.subset.groupby(by='frame') for cell in cells]

            n = 0
            drift_subset = np.zeros((sum(len(g) for g in groups_groups), 3))

            for groups in groups_groups:

                first_x = None
                first_y = None

                for frame, group in groups:
                    if first_x is None and first_y is None:
                        first_x, first_y = group.x.mean(), group.y.mean()
                    drift_subset[n, 0] = frame
                    drift_subset[n, 1] = group.x.mean() - first_x
                    drift_subset[n, 2] = group.y.mean() - first_y

                    n += 1

            print("XFIT")
            xfit = linregress(drift_subset[:, 0], drift_subset[:, 1])
            print(xfit)
            print("YFIT")
            yfit = linregress(drift_subset[:, 0], drift_subset[:, 2])
            print(yfit)

            for cell in cells:
                cell.subset.x -= cell.subset.frame * xfit.slope  # + xfit.intercept
                cell.subset.y -= cell.subset.frame * yfit.slope  # + yfit.intercept
        #
        #     #raise SystemExit

        if False and args.drift_correction:
            intermediate = data.groupby('frame')
            drift_subset = np.zeros((len(intermediate), 3))

            for n, (frame, g) in enumerate(intermediate):
                drift_subset[n, 0] = frame
                drift_subset[n, 1] = g.x.mean()
                drift_subset[n, 2] = g.y.mean()

            drift_subset[:, 1] -= drift_subset[0, 1]
            drift_subset[:, 2] -= drift_subset[0, 2]

            print("XFIT")
            xfit = linregress(drift_subset[:, 0], drift_subset[:, 1])
            print(xfit)
            print("YFIT")
            yfit = linregress(drift_subset[:, 0], drift_subset[:, 2])
            print(yfit)

            for cell in cells:
                cell.subset.x -= cell.subset.frame * xfit.slope + xfit.intercept
                cell.subset.y -= cell.subset.frame * yfit.slope + yfit.intercept
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
            Values.ListValue(None, [cell.name for cell in cells], 'cell'),
            Values.ListValue(None, [
                'custom_moving_brute', 'custom_moving_kd', 'custom_static_brute_locprec', 'custom_static_brute_sigma',
                'custom_static_kd_locprec', 'custom_static_kd_sigma', 'trackpy'
            ], 'tracker'),
            Values.IntValue('exposure_time', 86, minimum=0, maximum=1000, unit="ms"),
            Values.IntValue('calibration', int(1000 * args.calibration), minimum=0, maximum=1000, unit="nm·pixel⁻¹"),
            Values.FloatValue('maximum_displacement', 0.195, minimum=0, maximum=20, unit="µm"),
            Values.IntValue('maximum_blink_dark', 1, minimum=0, maximum=100, unit="frame(s)"),
            Values.IntValue('image_display_frame', 0, minimum=0, maximum=maximum_frame, unit="frame"),
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
        meshes = visuals.Mesh()

        plugin.view.add(scatter)
        plugin.view.add(lines)
        plugin.view.add(meshes)

        for n, cell in enumerate(cells):

            if not cell.hull.size:
                continue

            cell.contour_line = visuals.Line(pos=np.array([(x, y, -0.5) for x, y in cell.render_hull]),
                                             color=(0.0, 1.0, 0.0, 1.0), width=5)
            cell.contour_line_bordered = visuals.Line(pos=np.array([(x, y, -0.5) for x, y in cell.render_hull_bordered]),
                                             color=(0.0, 0.0, 1.0, 1.0), width=5)

            plugin.view.add(cell.contour_line)
            plugin.view.add(cell.contour_line_bordered)

            center = cv2.minAreaRect(cell.hull)[0]

            cell.text = visuals.Text(text='%d' % n, color=(1, 0, 0, 1), pos=np.array((center[0], center[1], 0.5)))

            plugin.view.add(cell.text)

            cell.render_mesh = contour_to_mesh(cell.contour, 0, maximum_frame)  # subset.frame.min(), subset.frame.max())

        result_table = [{} for _ in range(len(cells))]

        def _empty():
            for n, result in enumerate(result_table):
                result.update({
                    "Cell #": str(n),
                    "Max Displacement (µm)": float('nan'),
                    "Max Dark (frames)": float('nan'),
                    "D": float('nan'),
                    "D_MSD": float('nan'),
                    # "D_LAGT": float('nan'),
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
                    "Filenames Results full": '',
                    "Westerwalbesloh Ratio": float('nan')  # this needs to be renamed at some point ;)
                })

            self.output_model.update_data(result_table)

        _empty()

        precision_in_pixel = data.locprec.mean()
        sigma_in_pixel = data.sigma.mean()

        def _update(values):
            micron_per_pixel = (values.calibration / 1000.0)
            frames_per_second = 1000.0 / values.exposure_time

            precision_label.update(precision_in_pixel * micron_per_pixel)
            sigma_label.update(sigma_in_pixel * micron_per_pixel)

            def calc_diff(n):

                cell = cells[n]

                tracked = cell.tracked

                # had some discussions with slavko
                # if False:
                #     from matplotlib import pyplot
                #
                #     im = trackpy.imsd(tracked, micron_per_pixel, frames_per_second)
                #     fig, ax = pyplot.subplots()
                #     ax.plot(im.index, im, 'k-', alpha=0.01)
                #     ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
                #            xlabel='lag time $t$')
                #     ax.set_xscale('log')
                #     ax.set_yscale('log')
                #     fig.savefig('_test.pdf')

                # for pid, trajectory in tracked.groupby('particle'):
                #    msd = trackpy.msd(trajectory, micron_per_pixel, frames_per_second)
                #    pylab.plot(np.array(msd.index), np.array(msd))

                # imsd = trackpy.imsd(tracked, micron_per_pixel, frames_per_second)
                # print(imsd)

                emsd = trackpy.emsd(tracked, micron_per_pixel, frames_per_second)
                cell.emsd = emsd
                lagt = np.array(emsd.index)
                msd = np.array(emsd)

                dimensionality = 2  # we can observe only two dimensions, then it should be two?

                D = np.array(msd / (lagt * 2 * dimensionality))

                result_table[n]["D"] = D.mean()  # [0]
                # result_table[n]["D_MSD"] = msd[0]
                # result_table[n]["D_LAGT"] = lagt[0]

                # print("Cell", n)
                # print(np.c_[lagt, msd, D])

                self.output_model.update_data(result_table)

            def redo(n):

                cell = cells[n]
                subset = cell.subset

                before_tracking = time.time()

                if values.tracker == 'trackpy' and trackpy:
                    if (
                            'trackpy_tracked' in cell and
                            'last_memory' in cell and
                            cell.last_memory == values.maximum_blink_dark and
                            'last_displacement' in cell and
                            cell.last_displacement == values.maximum_displacement
                    ):
                        tracked = cell.trackpy_tracked
                    else:
                        tracked = trackpy.link_df(subset, values.maximum_displacement / micron_per_pixel,
                                                  memory=values.maximum_blink_dark, link_strategy='nonrecursive')
                        cell.last_memory = values.maximum_blink_dark
                        cell.last_displacement = values.maximum_displacement
                        cell.trackpy_tracked = tracked
                elif values.tracker.startswith('custom') and Tracker:

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
                        'custom_static_brute_locprec': (
                            tracker.STRATEGY_BRUTE_FORCE, tracker.TRACKING_STATIC, 'locprec'),
                        'custom_static_brute_sigma': (tracker.STRATEGY_BRUTE_FORCE, tracker.TRACKING_STATIC, 'sigma'),
                        'custom_moving_kd': (tracker.STRATEGY_KDTREE, tracker.TRACKING_MOVING, 'locprec'),
                        'custom_static_kd_locprec': (tracker.STRATEGY_KDTREE, tracker.TRACKING_STATIC, 'locprec'),
                        'custom_static_kd_sigma': (tracker.STRATEGY_KDTREE, tracker.TRACKING_STATIC, 'sigma')
                    }[values.tracker]

                    if what == 'locprec':
                        transfer['precision'] = subset.locprec
                    else:
                        transfer['precision'] = subset.sigma

                    tracker.track(transfer, values.maximum_displacement / micron_per_pixel, values.maximum_blink_dark,
                                  mode, strategy)

                    # np.save("_tmp.npy", transfer)

                    def my_emsd(data):
                        maxframe = 0
                        tracks = []
                        for label in sorted(np.unique(data['label'])):
                            trace = data[data['label'] == label].copy()
                            if len(trace) == 1:
                                continue
                            trace = np.sort(trace, order='frame')
                            relframe = trace['frame'].copy()
                            relframe -= relframe.min()
                            relframemax = relframe.max()
                            if relframemax > maxframe:
                                maxframe = relframemax
                            tracks.append((trace, relframe))

                        result = np.ones((len(tracks), maxframe + 1)) * float('nan')

                        for n, (trace, relframe) in enumerate(tracks):
                            for m, sqd in zip(relframe, trace['square_displacement']):
                                result[n, m] = sqd

                        y = np.nanmean(result, axis=0)

                        x = np.linspace(0, len(y) - 1, len(y))

                        Q = np.c_[x / frames_per_second, (y * (micron_per_pixel ** 2))][1:, :]

                        lagt = Q[:, 0]
                        msd = Q[:, 1]

                        dimensionality = 2  # we can observe only two dimensions, then it should be two?

                        D = np.array(msd / (lagt * 2 * dimensionality))
                        return D, Q

                    # D, Q = my_emsd(transfer)
                    result_table[n]["D"] = tracker.msd(transfer, micron_per_pixel, frames_per_second)  # D.mean()
                    # result_table[n]["D_MSD"] = float('nan')
                    # result_table[n]["D_MSD"] = (Q[:, 1]/Q[:, 0]).mean()

                    tracked = subset.copy()
                    tracked['particle'] = transfer['label']
                    # expect modern pandas!
                    tracked = tracked.sort_values(by=['particle', 'frame'])


                # if args.drift_correction:
                #    from trackpy import compute_drift, subtract_drift

                #    drift = compute_drift(tracked)
                #    print(drift)
                #    tracked = subtract_drift(drift)

                after_tracking = time.time()

                print("Tracking took: %.2fs" % (after_tracking-before_tracking))

                # tracked

                cell.tracked = tracked

                subset = tracked
                conn = np.array(tracked.particle, dtype=np.uint32)

                nconn = np.zeros(len(conn), dtype=np.bool)

                nconn[:-1] = conn[:-1] == conn[1:]

                conn = nconn

                # subset = data

                render_data = np.c_[subset.x, subset.y, subset.frame]

                cell.render_data = render_data
                cell.render_conn = conn

                scatter.set_data(render_data, edge_color=None, face_color=(1, 1, 1, 0.5), size=5)
                lines.set_data(render_data, color=(1, 1, 1, 0.5), connect=conn)

                meshes.set_data(cell.render_mesh, color=(1, 1, 1, 0.25))

                scatter.update()
                lines.update()
                meshes.update()

                plugin.add_turntable_camera()
                plugin.view.camera.depth_value = 1e9
                plugin.view.camera.fov = 0.0

                result_table[n]["Max Displacement (µm)"] = values.maximum_displacement
                result_table[n]["Max Dark (frames)"] = values.maximum_blink_dark

                result_table[n]["Filename AVG"] = os.path.basename(average_file)
                result_table[n]["Filenames Results"] = "::".join(os.path.basename(t) for t in tabular_files)
                result_table[n]["Filename AVG full"] = average_file
                result_table[n]["Filenames Results full"] = "::".join(tabular_files)

                # print(tracked)
                the_count = int(tracked.particle.max())
                result_table[n]["Count"] = the_count

                ###
                result_table[n]["Westerwalbesloh Ratio"] = the_count / len(tracked)
                ###

                result_table[n]["Mean Loc Precision (µm)"] = subset.locprec.mean() * micron_per_pixel
                result_table[n]["Mean Loc Sigma (µm)"] = subset.sigma.mean() * micron_per_pixel

                try:
                    smaller_axis = min(*cell.ellipse[1])
                    larger_axis = max(*cell.ellipse[1])
                except TypeError:
                    smaller_axis = larger_axis = 0.0

                result_table[n]["Ellipse small (µm)"] = smaller_axis * micron_per_pixel
                result_table[n]["Ellipse long (µm)"] = larger_axis * micron_per_pixel

                if len(cell.contour) > 0:
                    the_area = cv2.contourArea(cell.contour)
                    the_area *= micron_per_pixel ** 2
                    result_table[n]["Convex hull area (µm²)"] = the_area
                    result_table[n]["Count / µm²"] = the_count / the_area

                self.output_model.update_data(result_table)

            if values['clear']:
                _empty()

            if values.quit:
                raise SystemExit

            rendered_image.transform = STTransform(translate=(0.0, 0.0, values.image_display_frame))
            rendered_image.update()

            if values._modified == 'image_display_frame':
                return

            if values.refresh:
                try:
                    redo(int(values.cell.split(' ')[0]))
                except Exception as e:
                    print("error in cell", n, e)

            if values.msd:
                calc_diff(values.cell)

            if values.msd_all:
                for n in range(len(cells)):
                    calc_diff(n)
                print("Done.")

            if values.analyse_all or (values.tracker.startswith('custom') and values.live == 1):
                for n in range(len(cells)):
                    try:
                        redo(n)
                    except Exception as e:
                        print("error in cell", n, e)
                print("Done.")

                values.show_all = True

            if ('Everything' in values.cell) or ('Unassigned' in values.cell):
                values.show_all = False

            if values.show_all:

                try:

                    render_data = np.concatenate(
                        [cell.render_data for cell in cells if cell.render_data is not None])
                    colors = np.random.random((len(render_data), 4))
                    colors = (1, 1, 1, 0.5)
                    scatter.set_data(render_data, edge_color=None, face_color=colors, size=5)
                    scatter.update()

                    render_conn = np.concatenate(
                        [cell.render_conn for cell in cells if cell.render_conn is not None])
                    lines.set_data(render_data, color=(1, 1, 1, 0.5), connect=render_conn)
                    lines.update()

                except ValueError:
                    pass

            if values.show_all or ('Everything' in values.cell) or ('Unassigned' in values.cell):
                try:
                    render_mesh = np.concatenate(
                        [cell.render_mesh for cell in cells if cell.render_mesh is not None])
                    meshes.set_data(render_mesh, color=(1, 1, 1, 0.25))
                    meshes.update()
                except ValueError:
                    pass

        return _update


def main():
    SuperresolutionTracking.run()
