import os, sys
import glob
import pathlib
import itertools
import multiprocessing as mp

import numpy as np
import cv2

from ca_processing import GetContour, OverlayContour

from miv.core.pipeline import Pipeline


if __name__ == "__main__":
    # files = glob.glob('CalciumImaging/*/**.mkv')
    filename = "55Kprimary002.mkv"

    # tasks = itertools.product(files, thresholds)
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = pool.starmap(main, tasks)
    # print(results)

    # for path, thresholds in itertools.combinations(files, thresholds):
    #     main(path, detection_threshold=thresholds)

    get_contour = GetContour(filename=filename, area_threshold=750)
    overlay_contour = OverlayContour(filename=filename, kernel_size=13)
    get_contour >> overlay_contour
    Pipeline(overlay_contour).run(verbose=True)
