
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class DIPProject(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DIP Project - SEC Interactive Pipeline")
        self.geometry("1200x800")

        self.original_img = None
        self.current_img = None

        self.setup_gui()

    def setup_gui(self):
        self.sidebar = ctk.CTkScrollableFrame(self, width=250)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="DIP Filters Panel", font=("Arial", 20, "bold")).pack(pady=15)

        ctk.CTkButton(self.sidebar, text=" Load Image", command=self.load_image, fg_color="#2ECC71").pack(pady=5, fill="x")
        ctk.CTkButton(self.sidebar, text=" Reset Original", command=self.reset_image, fg_color="#E74C3C").pack(pady=5, fill="x")

        ctk.CTkLabel(self.sidebar, text="Point Processing", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        ctk.CTkButton(self.sidebar, text="Histogram Equalization", command=self.apply_hist_eq).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Log Transform", command=self.apply_log).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Gamma Brighten", command=lambda: self.apply_gamma(0.5)).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Gamma Darken", command=lambda: self.apply_gamma(2.0)).pack(pady=2, fill="x")

        ctk.CTkLabel(self.sidebar, text="Smoothing", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        ctk.CTkButton(self.sidebar, text="Average Blur", command=self.apply_mean_blur).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Gaussian Blur", command=self.apply_gaussian_blur).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Median Filter", command=self.apply_median_blur).pack(pady=2, fill="x")

        ctk.CTkLabel(self.sidebar, text="Sharpening", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        ctk.CTkButton(self.sidebar, text="Laplacian Filter", command=self.apply_laplacian).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Sobel Edge Detection", command=self.apply_sobel).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Unsharp Masking", command=self.apply_unsharp_mask).pack(pady=2, fill="x")

        ctk.CTkLabel(self.sidebar, text="Morphology", font=("Arial", 14, "bold")).pack(pady=(15, 5))
        ctk.CTkButton(self.sidebar, text="Erosion", command=self.apply_erosion).pack(pady=2, fill="x")
        ctk.CTkButton(self.sidebar, text="Dilation", command=self.apply_dilation).pack(pady=2, fill="x")

        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.img_label = ctk.CTkLabel(self.display_frame, text="Load Image to Start")
        self.img_label.pack(expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if path:
            try:
                img_array = np.fromfile(path, dtype=np.uint8)
                self.original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if self.original_img is not None:
                    self.current_img = self.original_img.copy()
                    self.update_display()
                else:
                    messagebox.showerror("Error", "Unable to load this image.")

            except Exception as e:
                messagebox.showerror("Loading Error", str(e))

    def reset_image(self):
        if self.original_img is not None:
            self.current_img = self.original_img.copy()
            self.update_display()

    def apply_hist_eq(self):
        if self.current_img is not None:
            img_gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            result = cv2.equalizeHist(img_gray)
            self.current_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.update_display()

    def apply_log(self):
        if self.current_img is not None:
            img_float = self.current_img.astype(np.float32)
            c = 255 / np.log(1 + np.max(img_float))
            log_image = c * np.log(1 + img_float)
            self.current_img = np.array(log_image, dtype=np.uint8)
            self.update_display()

    def apply_gamma(self, gamma_val):
        if self.current_img is not None:
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            norm = gray / 255.0
            result = np.power(norm, gamma_val)
            result = (result * 255).astype(np.uint8)
            self.current_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.update_display()

    def apply_mean_blur(self):
        if self.current_img is not None:
            self.current_img = cv2.blur(self.current_img, (5, 5))
            self.update_display()

    def apply_gaussian_blur(self):
        if self.current_img is not None:
            self.current_img = cv2.GaussianBlur(self.current_img, (5, 5), 0)
            self.update_display()

    def apply_median_blur(self):
        if self.current_img is not None:
            self.current_img = cv2.medianBlur(self.current_img, 5)
            self.update_display()

    def apply_laplacian(self):
        if self.current_img is not None:
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            result = cv2.convertScaleAbs(gray - lap)
            self.current_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            self.update_display()

    def apply_sobel(self):
        if self.current_img is not None:
            gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            mag = cv2.magnitude(sx, sy)
            mag = cv2.convertScaleAbs(mag)
            self.current_img = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
            self.update_display()

    def apply_unsharp_mask(self):
        if self.current_img is not None:
            blur = cv2.GaussianBlur(self.current_img, (5, 5), 0)
            self.current_img = cv2.addWeighted(self.current_img, 1.5, blur, -0.5, 0)
            self.update_display()

    def apply_erosion(self):
        if self.current_img is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.current_img = cv2.erode(self.current_img, kernel, iterations=1)
            self.update_display()

    def apply_dilation(self):
        if self.current_img is not None:
            kernel = np.ones((5, 5), np.uint8)
            self.current_img = cv2.dilate(self.current_img, kernel, iterations=1)
            self.update_display()

    def update_display(self):
        img_rgb = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((850, 700))
        img_tk = ImageTk.PhotoImage(img_pil)
        self.img_label.configure(image=img_tk, text="")
        self.img_label.image = img_tk


if __name__ == "__main__":
    app = DIPProject()
    app.mainloop()
