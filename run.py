import os, sys
import glob
import pathlib
import itertools
import multiprocessing as mp

import numpy as np
import cv2

def main(path, detection_threshold, show=False):

    path = pathlib.Path(path)
    output_path = path.stem + f'_output_{detection_threshold}.mp4'
    cap = cv2.VideoCapture(path.as_posix())
    ret, frame_origin = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not ret:
        print("Error: Could not open video.")
        cap.release()
        exit()

    # Setup to write processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_origin.shape[1], frame_origin.shape[0]))

    # List to store all contours
    all_contours = []
    previous_frame = None

    # First pass: Process video to accumulate contours
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # existing processing and contour detection logic here
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised_frame = cv2.fastNlMeansDenoising(gray_frame, None, 3, 7, 21)

        if previous_frame is not None:

            delta = cv2.absdiff(previous_frame, denoised_frame)
            thresh = cv2.threshold(delta, detection_threshold, 255, cv2.THRESH_BINARY)[1]

            some_threshold = 0.001

            # Add contours to the list  
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > some_threshold:  # filter out small changes
                    all_contours.append(contour)

            if show:
                concatnated_frame = cv2.vconcat([
                    cv2.hconcat([gray_frame, denoised_frame]),
                    cv2.hconcat([delta, thresh]),
                ])

                cv2.imshow('processing', concatnated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        previous_frame = denoised_frame   
    
    cv2.destroyAllWindows()

    # Create a blank image and draw contours on it
    contour_img = np.zeros_like(frame_origin, dtype=np.uint8)
    cv2.drawContours(contour_img, all_contours, -1, (255, 255, 255), 1)

    # Convert to grayscale
    contour_gray = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)

    # Apply erosion
    kernel = np.ones((7, 7), np.uint8)
    dilated_img = cv2.dilate(contour_gray, kernel, iterations = 1)

    # Find contours on the eroded image
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset video to the start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        out.write(frame)

        if show:
            cv2.imshow('Frame with Eroded Contours', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    files = glob.glob('CalciumImaging/*/**.mkv')
    thresholds = [5, 10, 25, 50]

    #print(files)
    #sys.exit()
    tasks = itertools.product(files, thresholds)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(main, tasks)
    print(results)

    # for path, thresholds in itertools.combinations(files, thresholds):
    #     main(path, detection_threshold=thresholds)