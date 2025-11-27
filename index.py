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

# ---------------- GUI FUNCTIONS --------------------

img_path = None

def select_image():
    global img_path
    img_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if img_path:
        img_preview = Image.open(img_path).convert("L")
        img_preview.thumbnail((200, 200))
        tk_img = ImageTk.PhotoImage(img_preview)
        img_label.config(image=tk_img)
        img_label.image = tk_img

def run_process():
    if not img_path:
        messagebox.showerror("Error", "Select an image first!")
        return

    try:
        sigma = float(sigma_entry.get())
        k = int(kernel_entry.get())
    except:
        messagebox.showerror("Error", "Enter valid numeric inputs.")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    noisy = add_gaussian_noise(img, sigma)
    gis = gradient_inverse_smoothing(noisy, k)
    mean_out = mean_filter_manual(noisy, k)

    mse_gis = mse(img, gis)
    mse_mean = mse(img, mean_out)

    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(out_dir+"/01_original.png", img)
    cv2.imwrite(out_dir+"/02_noisy.png", noisy)
    cv2.imwrite(out_dir+"/03_gradinv.png", gis)
    cv2.imwrite(out_dir+"/04_mean_filter.png", mean_out)

    montage = np.hstack([
        cv2.resize(img, (200,200)),
        cv2.resize(noisy, (200,200)),
        cv2.resize(gis, (200,200)),
        cv2.resize(mean_out, (200,200))
    ])
    cv2.imwrite(out_dir+"/montage.png", montage)

    # ------------------ GIF vs Mean comparison image ------------------
    gif_vs_mean = np.hstack([
        cv2.resize(gis, (300,300)),
        cv2.resize(mean_out, (300,300))
    ])

    cv2.imwrite(os.path.join(out_dir, "comparison_gif_vs_mean.png"), gif_vs_mean)


    # Saving MSEs
    with open(out_dir+"/results_summary.txt", "w") as f:
        f.write("Gradient Inverse Smoothing VS Simple Arithmetic Smoothing Results Summary\n")
        f.write("-------------------------------------------\n")
        f.write(f"Selected Image : {os.path.basename(img_path)}\n")
        f.write(f"Sigma (Noise)  : {sigma}\n")
        f.write(f"Kernel Size    : {k}\n\n")
        f.write(f"MSE (Gradient Inverse): {mse_gis:.4f}\n")
        f.write(f"MSE (Simple Mean):      {mse_mean:.4f}\n")
        f.write("-------------------------------------------\n")
        f.write("Files saved:\n")
        f.write(f"- original.png\n- noisy.png\n- gradinv_.png\n")
        f.write("- mean_filter.png\n- montage.png\n")
        

    cv2.imshow("Montage", montage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    messagebox.showinfo("Smoothing Completed",
        f"Saved results in folder:\n{out_dir}\n\n"
        f"MSE (Grad Inv): {mse_gis:.2f}\n"
        f"MSE (Mean): {mse_mean:.2f}"
    )

# ------------------- GUI LAYOUT ---------------------

root = Tk()
root.title("Gradient Inverse Smoothing Tool")

Label(root, text="Gradient Inverse Smoothing VS Simple Mean Smoothing", font=("Helvetica", 14, "bold")).pack(pady=10)

Button(root, text="Select Image", command=select_image, bg="#4CAF50", fg="white").pack(pady=5)

img_label = Label(root, width=200, height=200, bg="#ddd")
img_label.pack(pady=5)
img_label.pack_propagate(False)

frame = Frame(root)
frame.pack(pady=10)

Label(frame, text="Sigma:").grid(row=0, column=0)
sigma_entry = Entry(frame)
sigma_entry.insert(0, "25")
sigma_entry.grid(row=0, column=1)

Label(frame, text="Kernel size:").grid(row=1, column=0)
kernel_entry = Entry(frame)
kernel_entry.insert(0, "3")
kernel_entry.grid(row=1, column=1)

Button(root, text="Run", command=run_process).pack(pady=20)

root.mainloop()