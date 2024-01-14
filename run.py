import os, sys
import glob
import pathlib
import itertools
import multiprocessing as mp

import numpy as np
import cv2

from ca_processing import GetContour

from miv.core.pipeline import Pipeline


def overlay_contour(path, all_contours, mask_threshold=10, kernel_size=7):
    path = pathlib.Path(path)
    output_path = path.stem + f"_output.mp4"

    cap = cv2.VideoCapture(path.as_posix())
    ret, frame_origin = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (frame_origin.shape[1], frame_origin.shape[0])
    )

    if not ret:
        print("Error: Could not open video.")
        cap.release()
        exit()

    lower_white = np.array([0, 0, 0])
    upper_white = np.array([mask_threshold] * 3)
    mask = cv2.inRange(frame_origin, lower_white, upper_white)

    cv2.imwrite("mask.png", mask)

    filtered_contours = []

    # for contour in all_contours:
    #     # Create a blank image with the same dimensions as the mask
    #     contour_img = np.zeros_like(mask)
    #     # Draw the contour on the blank image
    #     cv2.drawContours(contour_img, [contour], -1, (255), thickness=cv2.FILLED)
    #     # Perform bitwise AND between contour image and mask
    #     intersection = cv2.bitwise_and(contour_img, mask)
    #     # Check if there is any overlap (non-zero pixels)
    #     if cv2.countNonZero(intersection) > 0:
    #         filtered_contours.append(contour)

    contour_img = np.zeros_like(frame_origin, dtype=np.uint8)
    cv2.drawContours(contour_img, all_contours, -1, (255, 255, 255), 1)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_img = cv2.dilate(contour_img, kernel, iterations=1)


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.drawContours(frame, filtered_contours, -1, (0, 255, 0), 2)

        frame_added = cv2.addWeighted(frame, 0.8, dilated_img, 0.2, 0)
        out.write(frame_added)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # files = glob.glob('CalciumImaging/*/**.mkv')
    filename = "sample.mp4"
    thresholds = [5, 10]

    # tasks = itertools.product(files, thresholds)
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = pool.starmap(main, tasks)
    # print(results)

    # for path, thresholds in itertools.combinations(files, thresholds):
    #     main(path, detection_threshold=thresholds)

    get_contour = GetContour(filename=filename)
    Pipeline(get_contour).run(verbose=True)

    all_contours = get_contour.output()
    overlay_contour(filename, all_contours)