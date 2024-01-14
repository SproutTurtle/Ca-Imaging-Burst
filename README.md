<div align="center">
 <h2> Calcium-Imaging video post-processing and burst detection </h2>
</div>

## How to use

```py
from ca_processing import GetContour, OverlayContour
from miv.core.pipeline import Pipeline


def main(): 
    get_contour = GetContour(filename=filename, area_threshold=750, starting_frame=500, bkg_history=200)
    overlay_contour = OverlayContour(filename=filename, kernel_size=13)
    get_contour >> overlay_contour
    Pipeline(overlay_contour).run(verbose=True)

if __name__ == "__main__":
    filename = "sample.mp4"
    main(filename)

```