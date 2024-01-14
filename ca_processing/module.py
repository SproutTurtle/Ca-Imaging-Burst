__all__ = ["GetContour"]

import os, sys
import glob
import pathlib
import itertools
import multiprocessing as mp

import numpy as np
import cv2

from dataclasses import dataclass

from miv.core.operator import OperatorMixin
from miv.core.pipeline import Pipeline
from miv.core.operator import cache_call


@dataclass
class GetContour(OperatorMixin):
    path: str
    area_threshold: int = 100
    tag: str = "get contour"

    def __post_init__():
        super().__init__()

    @cache_call
    def __call__(self):
        path = pathlib.Path(self.path)
        cap = cv2.VideoCapture(path.as_posix())
        output_path = path.stem + f"_processing.mp4"
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

        back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        all_contours = []

        # Process video to accumulate contours
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fgMask = back_sub.apply(frame)

            # existing processing and contour detection logic here
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            denoised_frame = cv2.fastNlMeansDenoising(gray_frame, None, 3, 7, 21)

            # Add contours to the list
            contours, _ = cv2.findContours(
                fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if cv2.contourArea(contour) > self.area_threshold:
                    all_contours.append(contour)

            concatnated_frame = cv2.vconcat(
                [
                    cv2.hconcat([gray_frame, denoised_frame]),
                ]
            )

            out.write(concatnated_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return all_contours
