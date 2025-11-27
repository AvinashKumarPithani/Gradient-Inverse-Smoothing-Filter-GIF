Gradient Inverse Smoothing Filter Project
=========================================

1. Project Description
----------------------
This project implements the Gradient Inverse Smoothing Filter (GIF) manually. The objective is to denoise an image while preserving important edges, and to compare the result against simple arithmetic smoothing.

The project includes:
- Addition of zero-mean Gaussian noise
- Manual arithmetic mean filter (simple averaging)
- Gradient Inverse Smoothing Filter (GIF)
- Computation of Mean Squared Error (MSE)
- Tkinter-based graphical user interface for easy use
- Output image generation and comparison figures

The GIF follows the professor’s formula:

    g(i,j) = p * I_center + Σ( h_n * I_neighbor )

Where:
    h_n = (1 - p) * (u_n / Σu)
    u_n = 1 / |I_neighbor - I_center|
    If neighbor == center: u_n = p

No OpenCV built-in smoothing functions such as blur, GaussianBlur, or bilateralFilter are used.
All smoothing operations are implemented manually as required.

---------------------------------------------------

2. Folder Structure
-------------------
ImageDenoising/
│
├── images/                     # Input images provided by the user
│       └── sample1.png
│
├── results/                    # Auto-generated output files
│       ├── 01_original.png
│       ├── 02_noisy.png
│       ├── 03_gradinv.png
│       ├── 04_mean_filter.png
│       ├── montage.png
│       ├── comparison_gif_vs_mean.png
│       ├── comparison_noisy_gif_mean.png   (optional)
│       └── results_summary.txt
│
├── filters.py                  # Core algorithm functions (noise, GIF, mean, MSE)
└── index.py                    # GUI and program execution

---------------------------------------------------

3. Input Data Usage
--------------------
Place any grayscale or color image into the "images" folder.

Supported formats:
- .png
- .jpg
- .jpeg
- .bmp

Images are automatically converted to grayscale before processing.

---------------------------------------------------

4. Parameters Used
-------------------

Sigma (σ):
    - Standard deviation of Gaussian noise
    - Controls the strength of noise added to the input
    - Example: σ = 25

Kernel Size (k):
    - Size of the sliding window (must be odd: 3, 5, 7…)
    - Used in both GIF and mean filter

GIF Parameter (p):
    - Weight given to the center pixel
    - Default: p = 0.2
    - Lower p ⇒ stronger smoothing, higher p ⇒ more detail preserved

---------------------------------------------------

5. Output Description
----------------------

(1) 01_original.png  
      The original grayscale image.

(2) 02_noisy.png  
      Original + Gaussian noise:
          noisy = original + N(0, σ)
      Values clipped to [0, 255].

(3) 03_gradinv.png  
      Output of the Gradient Inverse Filter.
      Characteristics:
        - Edge-preserving smoothing
        - Neighbors are weighted inversely to intensity difference

(4) 04_mean_filter.png  
      Output of manual arithmetic mean filter.
      Characteristics:
        - Strong smoothing
        - Edge blurring

(5) montage.png  
      Side-by-side comparison:
        [Original | Noisy | GIF | Mean Filter]

(6) comparison_gif_vs_mean.png  
      Direct comparison:
        [GIF | Mean Filter]

(7) comparison_noisy_gif_mean.png (optional)  
      Three-panel comparison:
        [Noisy | GIF | Mean Filter]

(8) results_summary.txt  
      Contains:
          MSE (Gradient Inverse): XXXXX
          MSE (Simple Mean):      XXXXX

---------------------------------------------------

6. How to Run the Project
--------------------------

Run GUI:
    python index.py

Steps:
    1. Click “Select Image”
    2. Choose an image from /images
    3. Enter sigma and kernel size
    4. Click “Run”
    5. Results are saved to the /results folder

---------------------------------------------------

7. Dependencies
----------------
Install the required Python packages:

    pip install numpy opencv-python pillow

---------------------------------------------------

8. Notes
---------
- No OpenCV built-in filters are used (as per project rules).
- All gradient, noise, and smoothing operations are manually implemented.
- GIF implementation strictly follows all required algorithmic constraints.
- Output images are saved automatically for documentation and presentation.

