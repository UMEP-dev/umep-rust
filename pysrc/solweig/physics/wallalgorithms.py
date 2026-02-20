__author__ = "xlinfr"

import math

import numpy as np

from ..progress import get_progress_iterator
from .morphology import rotate_array


def findwalls(a, walllimit):
    # This function identifies walls based on a DSM and a wall-height limit
    # Walls are represented by outer pixels within building footprints
    #
    # Fredrik Lindberg, Goteborg Urban Climate Group
    # fredrikl@gvc.gu.se
    # 20150625
    #
    # For each pixel, find the max of its 4 cardinal neighbors (cross kernel).
    # Wall height = max_neighbor - self, clipped to walllimit.

    walls = np.zeros_like(a, dtype=np.float32)

    # Max of 4 cardinal neighbors for all interior pixels
    max_neighbors = np.maximum.reduce(
        [
            a[:-2, 1:-1],  # north
            a[2:, 1:-1],  # south
            a[1:-1, :-2],  # west
            a[1:-1, 2:],  # east
        ]
    )
    walls[1:-1, 1:-1] = max_neighbors

    walls = walls - a
    walls[walls < walllimit] = 0

    # Zero borders
    walls[0, :] = 0
    walls[-1, :] = 0
    walls[:, 0] = 0
    walls[:, -1] = 0

    return walls


def filter1Goodwin_as_aspect_v3(walls, scale, a, feedback=None):
    """
    tThis function applies the filter processing presented in Goodwin et al (2010) but instead for removing
    linear fetures it calculates wall aspect based on a wall pixels grid, a dsm (a) and a scale factor

    Fredrik Lindberg, 2012-02-14
    fredrikl@gvc.gu.se

    Translated: 2015-09-15

    :param walls:
    :param scale:
    :param a:
    :return: dirwalls
    """
    # Try Rust implementation first (much faster)
    try:
        import threading

        from ..progress import ProgressReporter
        from ..rustalgos import wall_aspect as _wa_rust

        walls_f32 = np.asarray(walls, dtype=np.float32)
        dsm_f32 = np.asarray(a, dtype=np.float32)

        runner = _wa_rust.WallAspectRunner()
        result = [None]
        error = [None]

        def _run():
            try:
                result[0] = runner.compute(walls_f32, float(scale), dsm_f32)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        # Poll progress (180 angle iterations)
        total = 180
        pbar = ProgressReporter(total=total, desc="Computing wall aspects", feedback=feedback)
        last = 0
        while thread.is_alive():
            thread.join(timeout=0.05)
            done = runner.progress()
            if done > last:
                pbar.update(done - last)
                last = done
            # Check QGIS cancellation
            if feedback is not None and hasattr(feedback, "isCanceled") and feedback.isCanceled():
                runner.cancel()
                thread.join(timeout=5.0)
                pbar.close()
                return np.zeros_like(walls_f32)
        if last < total:
            pbar.update(total - last)
        pbar.close()

        thread.join()
        if error[0] is not None:
            raise error[0]
        return np.asarray(result[0])
    except ImportError:
        pass

    # Python fallback
    row = a.shape[0]
    col = a.shape[1]

    filtersize = np.floor((scale + 0.0000000001) * 9)
    if filtersize <= 2:
        filtersize = 3
    elif filtersize != 9 and filtersize % 2 == 0:
        filtersize = filtersize + 1

    filthalveceil = int(np.ceil(filtersize / 2.0))
    filthalvefloor = int(np.floor(filtersize / 2.0))

    filtmatrix = np.zeros((int(filtersize), int(filtersize)), dtype=np.float32)
    buildfilt = np.zeros((int(filtersize), int(filtersize)), dtype=np.float32)

    filtmatrix[:, filthalveceil - 1] = 1
    n = filtmatrix.shape[0] - 1
    buildfilt[filthalveceil - 1, 0:filthalvefloor] = 1
    buildfilt[filthalveceil - 1, filthalveceil : int(filtersize)] = 2

    y = np.zeros((row, col), dtype=np.float32)  # final direction
    z = np.zeros((row, col), dtype=np.float32)  # temporary direction
    x = np.zeros((row, col), dtype=np.float32)  # building side
    walls[walls > 0.5] = 1

    for h in get_progress_iterator(
        range(0, 180), desc="Computing wall aspects", feedback=feedback
    ):  # =0:1:180 #%increased resolution to 1 deg 20140911
        filtmatrix1temp = rotate_array(filtmatrix, h, order=1, reshape=False, mode="nearest")  # bilinear
        filtmatrix1 = np.round(filtmatrix1temp)
        filtmatrixbuildtemp = rotate_array(buildfilt, h, order=0, reshape=False, mode="nearest")  # Nearest neighbor
        # filtmatrixbuild = np.round(filtmatrixbuildtemp / 127.)
        filtmatrixbuild = np.round(filtmatrixbuildtemp)
        index = 270 - h
        if h == 150:
            filtmatrixbuild[:, n] = 0
        if h == 30:
            filtmatrixbuild[:, n] = 0
        if index == 225:
            # n = filtmatrix.shape[0] - 1  # length(filtmatrix);
            filtmatrix1[0, 0] = 1
            filtmatrix1[n, n] = 1
        if index == 135:
            # n = filtmatrix.shape[0] - 1  # length(filtmatrix);
            filtmatrix1[0, n] = 1
            filtmatrix1[n, 0] = 1

        for i in range(int(filthalveceil) - 1, row - int(filthalveceil) - 1):  # i=filthalveceil:sizey-filthalveceil
            for j in range(
                int(filthalveceil) - 1, col - int(filthalveceil) - 1
            ):  # (j=filthalveceil:sizex-filthalveceil
                if walls[i, j] == 1:
                    wallscut = (
                        walls[
                            i - filthalvefloor : i + filthalvefloor + 1,
                            j - filthalvefloor : j + filthalvefloor + 1,
                        ]
                        * filtmatrix1
                    )
                    dsmcut = a[
                        i - filthalvefloor : i + filthalvefloor + 1,
                        j - filthalvefloor : j + filthalvefloor + 1,
                    ]
                    if z[i, j] < wallscut.sum():  # sum(sum(wallscut))
                        z[i, j] = wallscut.sum()  # sum(sum(wallscut));
                        if np.sum(dsmcut[filtmatrixbuild == 1]) > np.sum(dsmcut[filtmatrixbuild == 2]):
                            x[i, j] = 1
                        else:
                            x[i, j] = 2

                        y[i, j] = index

    y[(x == 1)] = y[(x == 1)] - 180
    y[(y < 0)] = y[(y < 0)] + 360

    grad, asp = get_ders(a, scale)

    y = y + ((walls == 1) * 1) * ((y == 0) * 1) * (asp / (math.pi / 180.0))

    dirwalls = y

    return dirwalls


def cart2pol(x, y, units="deg"):
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if units in ["deg", "degs"]:
        theta = theta * 180 / np.pi
    return theta, radius


def get_ders(dsm, scale):
    # dem,_,_=read_dem_grid(dem_file)
    dx = 1 / scale
    # dx=0.5
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, "rad")
    grad = np.arctan(grad)
    asp = asp * -1
    asp = asp + (asp < 0) * (np.pi * 2)
    return grad, asp
