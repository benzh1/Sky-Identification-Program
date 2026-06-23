# Sky Identification

A computer vision program that segments sky regions in video footage, available in both Python and C++. For each frame, it produces a segmented overlay highlighting detected sky areas alongside a probability map showing the confidence of sky classification across the frame.

## Overview

Given a video file as input, the program processes it frame by frame, applying image segmentation techniques to identify which regions of each frame belong to the sky. Two outputs are generated: an annotated video with sky regions marked, and a probability heatmap video that visualises how confidently each area of the frame is classified as sky. The project is implemented in both Python and C++ to demonstrate the same solution across different levels of the stack.

## Features

- **Frame-by-frame sky segmentation** — processes each frame of the input video to identify sky regions
- **Probability map output** — generates a heatmap video showing the likelihood of sky classification per pixel region
- **Dual implementation** — available in both Python and C++ for flexibility and performance comparison
- **Video I/O** — accepts `.avi` video input and produces annotated video output

## Output

Running the program on an input video produces two output files:

- **Segmented video** — the original footage with sky regions highlighted per frame
- **Probability map video** — a heatmap visualisation showing confidence scores across each frame

## Technologies

- **Python / C++** — dual implementation of the same segmentation pipeline
- **OpenCV** — video I/O, frame processing, and image segmentation
- **NumPy** — pixel-level array operations (Python version)
- **Computer vision** — colour-space analysis and probabilistic classification for sky region detection
