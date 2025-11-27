import numpy as np
import cv2
import os
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from filters import (
    add_gaussian_noise,
    mean_filter_manual,
    gradient_inverse_smoothing,
    mse
)

img_path = None
img_label = None
sigma_entry = None
kernel_entry = None

# ---------------- GUI / Utility functions --------------------

def select_image():
    """Open file dialog and show a thumbnail in the GUI."""
    global img_path, img_label
    img_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if img_path:
        try:
            img_preview = Image.open(img_path).convert("L")
            img_preview.thumbnail((200, 200))
            tk_img = ImageTk.PhotoImage(img_preview)
            img_label.config(image=tk_img)
            img_label.image = tk_img
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
            img_path = None


def add_separator(height, thickness=10, color=255):
    """Creates a vertical blank separator between images (uint8)."""
    return np.full((height, thickness), color, dtype=np.uint8)


def run_process():
    """Main processing pipeline called from GUI."""
    global img_path, sigma_entry, kernel_entry

    if not img_path:
        messagebox.showerror("Error", "Select an image first!")
        return

    # read numeric parameters
    try:
        sigma = float(sigma_entry.get())
        k = int(kernel_entry.get())
        if k % 2 == 0 or k < 1:
            messagebox.showerror("Error", "Kernel size must be an odd positive integer (3,5,7...).")
            return
    except Exception:
        messagebox.showerror("Error", "Enter valid numeric values for Sigma and Kernel size.")
        return

    # read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        messagebox.showerror("Error", "Cannot read the selected image.")
        return

    # processing
    noisy = add_gaussian_noise(img, sigma)
    gis = gradient_inverse_smoothing(noisy, k)
    mean_out = mean_filter_manual(noisy, k)

    mse_gis = mse(img, gis)
    mse_mean = mse(img, mean_out)

    # prepare output folder
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    # Save basic outputs
    cv2.imwrite(os.path.join(out_dir, "01_original.png"), img)
    cv2.imwrite(os.path.join(out_dir, "02_noisy.png"), noisy)
    cv2.imwrite(os.path.join(out_dir, "03_gradinv.png"), gis)
    cv2.imwrite(os.path.join(out_dir, "04_mean_filter.png"), mean_out)

    # ---------------------- Creating Montage with Separators ------------------------
    h = 200
    sep = add_separator(h, thickness=15, color=255)  # vertical white separator

    img_res   = cv2.resize(img, (200,200))
    noisy_res = cv2.resize(noisy, (200,200))
    gis_res   = cv2.resize(gis, (200,200))
    mean_res  = cv2.resize(mean_out, (200,200))

    montage = np.hstack([
        img_res, sep,
        noisy_res, sep,
        gis_res, sep,
        mean_res
    ])

    cv2.imwrite(os.path.join(out_dir, "montage.png"), montage)

    # ------------------ GIF vs Mean comparison (with separator) ------------------
    h2 = 300
    sep2 = add_separator(h2, thickness=20, color=255)

    gif_vs_mean = np.hstack([
        cv2.resize(gis, (300,300)),
        sep2,
        cv2.resize(mean_out, (300,300))
    ])

    cv2.imwrite(os.path.join(out_dir, "comparison_gif_vs_mean.png"), gif_vs_mean)

    # ------------------ Save MSE Summary ------------------
    summary_path = os.path.join(out_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Gradient Inverse Smoothing VS Simple Arithmetic Smoothing Results Summary\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write(f"Selected Image : {os.path.basename(img_path)}\n")
        f.write(f"Sigma (Noise)  : {sigma}\n")
        f.write(f"Kernel Size    : {k}\n\n")
        f.write(f"MSE (Gradient Inverse): {mse_gis:.4f}\n")
        f.write(f"MSE (Simple Mean):      {mse_mean:.4f}\n")
        f.write("-----------------------------------------------------------------------\n")
        f.write("Files saved:\n")
        f.write("- 01_original.png\n")
        f.write("- 02_noisy.png\n")
        f.write("- 03_gradinv.png\n")
        f.write("- 04_mean_filter.png\n")
        f.write("- montage.png\n")
        f.write("- comparison_gif_vs_mean.png\n")
        f.write("\n")

    # showing the montage window (works on local desktop)
    try:
        cv2.imshow("Montage", montage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        # In environments without display (e.g., remote server), skip imshow
        pass

    messagebox.showinfo(
        "Smoothing Completed",
        f"Saved results in folder:\n{out_dir}\n\n"
        f"MSE (Grad Inv): {mse_gis:.2f}\n"
        f"MSE (Mean): {mse_mean:.2f}"
    )


# ------------------- Build GUI Layout ---------------------

root = Tk()
root.title("Gradient Inverse Smoothing Tool")

Label(root, text="Gradient Inverse Smoothing VS Simple Mean Smoothing", font=("Helvetica", 14, "bold")).pack(pady=10)

Button(root, text="Select Image", command=select_image, bg="#4CAF50", fg="white").pack(pady=5)

img_label = Label(root, width=200, height=200, bg="#ddd")
img_label.pack(pady=5)
img_label.pack_propagate(False)

frame = Frame(root)
frame.pack(pady=10)

Label(frame, text="Sigma:").grid(row=0, column=0, sticky="e", padx=4, pady=2)
sigma_entry = Entry(frame, width=10)
sigma_entry.insert(0, "25")
sigma_entry.grid(row=0, column=1, padx=4, pady=2)

Label(frame, text="Kernel size:").grid(row=1, column=0, sticky="e", padx=4, pady=2)
kernel_entry = Entry(frame, width=10)
kernel_entry.insert(0, "3")
kernel_entry.grid(row=1, column=1, padx=4, pady=2)

Button(root, text="Run", command=run_process, bg="#2196F3", fg="white").pack(pady=20)

# Info label
Label(root, text="Results are saved in './results'", fg="gray").pack(pady=5)

root.mainloop()