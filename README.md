# Hierarchical Uncertainty Aggregation

This repo implements hierarchical uncertainty aggregation from the paper _Active Learning for Object Detection with Evidential Deep Learning and Hierarchical Uncertainty Aggregation_ by Park et al. (2022).

It is in written in Rust with bindings for Python.

## Build

Run `maturin build` to build the wheel.

## Usage

1. Import the package with `import hua`.
2. Create a `hua.HUA` object with `iou_threshold` and `score_threshold`.
3. Call `hua.HUA.run()` with:

    ```text
    bounding_boxes: Array of bounding boxes in the format [x1, y1, x2, y2] of shape (N, 4),
    class_probabilities: Array of class probabilities of shape (N, C),
    uncertainty_scores: Array of instance level uncertainty scores of shape (N,),
    scales: Array of FPN scales from which the corresponding prediction was obtained of shape (N,).
    ```

4. The method returns the informativeness score of the image and the indices of the groups created by HUA.

## Acknowledgements

[Powerboxes](https://github.com/Smirkey/powerboxes): It was used to speed up the IOU calculation in the code.
