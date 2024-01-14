__all__ = ["GetContour", "OverlayContour"]

import os
import pathlib

import numpy as np
import cv2

from dataclasses import dataclass

from miv.core.operator import OperatorMixin
from miv.core.operator.wrapper import cache_call


@dataclass
class GetContour(OperatorMixin):
    """GetContour

    Parameter:
    ----------
    TODO
    """

    filename: str
    area_threshold: int = 300
    bkg_history: int = 500
    bkg_var_threshold: int = 16
    starting_frame: int = 0
    tag: str = "get contour"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self):
        path = pathlib.Path(self.filename)
        cap = cv2.VideoCapture(path.as_posix())
        if self.starting_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.starting_frame)

        output_path = os.path.join(self.analysis_path, path.stem + f"_processing.mp4")
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
            history=self.bkg_history,
            varThreshold=self.bkg_var_threshold,
            detectShadows=False,
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


@dataclass
class OverlayContour(OperatorMixin):
    """OverlayContour

    Parameter:
    ----------
    TODO
    """

    filename: str
    mask_threshold: int = 10
    kernel_size: int = 7

    tag: str = "overlay contour"

    def __post_init__(self):
        super().__init__()

    @cache_call
    def __call__(self, all_contours):
        path = pathlib.Path(self.filename)
        output_path = os.path.join(self.analysis_path, path.stem + f"_output.mp4")

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
        upper_white = np.array([self.mask_threshold] * 3)
        mask = cv2.inRange(frame_origin, lower_white, upper_white)

        contour_img = np.zeros_like(frame_origin, dtype=np.uint8)
        cv2.drawContours(contour_img, all_contours, -1, (255, 255, 255), 1)

        # Dilate
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        dilated_img = cv2.dilate(contour_img, kernel, iterations=1)

        # Masking
        dilated_img = cv2.bitwise_and(dilated_img, dilated_img, mask=~mask)
        mask = np.zeros_like(mask)
        mask[5:-5, 5:-5, ...] = 255
        dilated_img = cv2.bitwise_and(dilated_img, dilated_img, mask=mask)

        contours, _ = cv2.findContours(
            cv2.cvtColor(dilated_img, cv2.COLOR_BGR2GRAY),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.addWeighted(frame, 0.8, dilated_img, 0.2, 0)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return mask

    def plot_mask(self, outputs, inputs, show=False, save_path=None):
        cv2.imwrite(
            os.path.join(save_path, "mask.png"),
            outputs,
        )
