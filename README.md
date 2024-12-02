# Bamboleo Detection

Bamboleo is a balancing game where players take turns removing wooden pieces from a precariously balanced platform without making it tip over. This project aims to detect the wooden block using computer vision.

## Table of Contents
- [Bamboleo Detection](#bamboleo-detection)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Technical Implementation Details](#technical-implementation-details)
    - [1. Color-Based Board Detection](#1-color-based-board-detection)
    - [2. Board Mask Generation](#2-board-mask-generation)
    - [3. Binary Image Inversion](#3-binary-image-inversion)
    - [4. Mask Application](#4-mask-application)
    - [5. Piece Detection and Sorting](#5-piece-detection-and-sorting)
- [Result](#result)

## Installation
To install the necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To use the bamboleo detection code, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/bamboleo-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd bamboleo-detection
    ```
3. Run the detection script:
    ```bash
    python scripts/detect.py --video video/sample_video.mov --config
    ```

## Technical Implementation Details

### 1. Color-Based Board Detection
The system first isolates the Bamboleo game board using color thresholding techniques:

* Converts the input frame from BGR to HSV color space for more robust color detection
* Applies yellow color filtering using specific HSV range values
<p align="center" width="50%">
    <img width="50%" src=resources/color_trackbar.png>
</p>

* Creates a binary mask where white pixels represent yellow regions (the game board)
* This step effectively separates the distinctive yellow Bamboleo platform from the background
<p align="center" width="50%">
    <img width="50%" src=resources/1_yellow_binary.png>
</p>

### 2. Board Mask Generation
The yellow board mask is refined through:

* Morphological operations to remove noise and fill gaps
* Contour detection to find the largest yellow region
* Creation of a clean mask representing only the game board area
* This mask serves as the region of interest for subsequent processing
<p align="center" width="50%">
    <img width="50%" src=resources/2_yellow_mask.png>
</p>

### 3. Binary Image Inversion
To isolate the wooden pieces:

* The yellow binary image is inverted using bitwise NOT operation
* This creates a negative image where the board becomes black
* Game pieces and other objects become white
* This step prepares for piece detection by creating contrast between pieces and board
<p align="center" width="50%">
    <img width="50%" src=resources/3_inv_yellow_binary.png>
</p>


### 4. Mask Application
Combines the processed images using bitwise operations:

* Performs bitwise AND between yellow mask and inverted binary frame
* This operation isolates only the wooden pieces that are on the game board
* Eliminates detection of objects outside the game area
* Results in a clean binary image showing only the pieces
<p align="center" width="50%">
    <img width="50%" src=resources/4_object_frame.png>
</p>

### 5. Piece Detection and Sorting
Identifies individual game pieces through:

* Contour detection on the masked binary image
* Filtering contours based on minimum area to remove noise
* Calculation of piece centroids
* Computing Euclidean distances from each piece to board center
* Sorting pieces by distance for strategic analysis
* This provides a prioritized list of pieces based on their position relative to the balance point
<p align="center" width="50%">
    <img width="50%" src=resources/5_result_frame.png>
</p>

# Result 
<p align="center" width="50%">
    <img width="50%" src=resources/output.gif>
</p>