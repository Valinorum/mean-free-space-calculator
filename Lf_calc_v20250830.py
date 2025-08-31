import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.ndimage import label
import os
import csv
# Matplotlib is now a required dependency for the Path object
#from matplotlib.path import Path

class MeanPathApp:
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Mean Free Path Calculator")
        self.root.geometry("1200x800")

        # --- State Variables ---
        self.original_image = None
        self.processed_image = None
        self.largest_component_mask = None
        self.full_binary_mask = None # To store the mask with all components
        self.polygon_points = []
        self.is_drawing_polygon = False
        self.is_picking_pixel = False
        self.picked_point_canvas_coords = None
        self.image_path = None # No default image
        self.batch_results = []

        # --- GUI Layout ---
        main_frame = tk.Frame(root, bg="gray90")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_panel = tk.Frame(main_frame, width=350, bg="white", bd=2, relief=tk.RIDGE)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        self.canvas = tk.Canvas(main_frame, bg="gray70", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.welcome_message = self.canvas.create_text(450, 400, text="Open an image or folder to begin analysis.", font=("Helvetica", 16), fill="white")

        # --- Widgets for Control Panel ---
        tk.Label(control_panel, text="Controls", font=("Helvetica", 16, "bold"), bg="white").pack(pady=10, padx=10, anchor="w")

        # 1. Image Loading
        load_frame = tk.LabelFrame(control_panel, text="1. Load Data", padx=10, pady=10, bg="white")
        load_frame.pack(pady=5, padx=10, fill=tk.X)
        tk.Button(load_frame, text="Open Single Image", command=self.load_image).pack(fill=tk.X, pady=2)
        tk.Button(load_frame, text="Analyze Folder (Batch)", command=self.load_folder).pack(fill=tk.X, pady=2)

        # 2. Scale
        scale_frame = tk.LabelFrame(control_panel, text="2. Set Real-World Scale (µm)", padx=10, pady=10, bg="white")
        scale_frame.pack(pady=5, padx=10, fill=tk.X)
        self.width_var = tk.StringVar(value="100")
        tk.Entry(scale_frame, textvariable=self.width_var).pack(fill=tk.X)

        # 3. Processing Settings
        proc_settings_frame = tk.LabelFrame(control_panel, text="3. Processing Settings", padx=10, pady=10, bg="white")
        proc_settings_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.threshold_var = tk.IntVar(value=150)
        self.threshold_label = tk.Label(proc_settings_frame, text=f"Threshold Value: {self.threshold_var.get()}", bg="white")
        self.threshold_label.pack()
        tk.Scale(proc_settings_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.threshold_var, command=self.update_threshold_display).pack(fill=tk.X)
        
        self.invert_var = tk.BooleanVar()
        tk.Checkbutton(proc_settings_frame, text="Material is Lighter (Invert)", variable=self.invert_var, bg="white", command=self.update_image_processing).pack(anchor='w')
        
        self.show_all_voids_var = tk.BooleanVar(value=False)
        tk.Checkbutton(proc_settings_frame, text="Show All Void Areas (Visual Aid)", variable=self.show_all_voids_var, bg="white", command=self.update_image_processing).pack(anchor='w')

        # 4. Manual Threshold Sampler
        poly_frame = tk.LabelFrame(control_panel, text="4. Manual Threshold Sampler (Optional)", padx=10, pady=10, bg="white")
        poly_frame.pack(pady=5, padx=10, fill=tk.X)
        self.poly_button = tk.Button(poly_frame, text="Start Sampling", command=self.toggle_polygon_drawing)
        self.poly_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        tk.Button(poly_frame, text="Reset", command=self.reset_polygon).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # 5. Test Pixel Value
        test_pixel_frame = tk.LabelFrame(control_panel, text="5. Test Pixel Value", padx=10, pady=10, bg="white")
        test_pixel_frame.pack(pady=5, padx=10, fill=tk.X)
        self.pixel_pick_button = tk.Button(test_pixel_frame, text="Pick Pixel", command=self.toggle_pixel_picking)
        self.pixel_pick_button.pack(side=tk.LEFT, padx=(0, 10))
        self.pixel_value_var = tk.StringVar(value="Value: N/A")
        tk.Label(test_pixel_frame, textvariable=self.pixel_value_var, bg="white").pack(side=tk.LEFT)

        # 6. Calculation Settings
        calc_settings_frame = tk.LabelFrame(control_panel, text="6. Calculation Settings", padx=10, pady=10, bg="white")
        calc_settings_frame.pack(pady=5, padx=10, fill=tk.X)
        
        tk.Label(calc_settings_frame, text="Analysis Method:", bg="white").pack(anchor='w')
        self.method_var = tk.StringVar(value="Free Space (Distance Transform)")
        method_options = ["Free Space (Largest Square)", "Free Space (Distance Transform)", "Free Space (Box Counting)", "Mean Free Path (Line Intercept)"]
        tk.OptionMenu(calc_settings_frame, self.method_var, *method_options, command=self.on_method_change).pack(fill=tk.X)

        self.analyze_all_var = tk.BooleanVar(value=False)
        self.analyze_all_cb = tk.Checkbutton(calc_settings_frame, text="Analyze all components (Sanity Check)", variable=self.analyze_all_var, bg="white")
        self.analyze_all_cb.pack(anchor='w', pady=(5,0))

        tk.Label(calc_settings_frame, text="Random Line/Box Count:", bg="white").pack(anchor='w', pady=(5,0))
        self.lines_var = tk.StringVar(value="5000")
        self.lines_entry = tk.Entry(calc_settings_frame, textvariable=self.lines_var)
        self.lines_entry.pack(fill=tk.X)
        self.on_method_change() # Set initial state

        # 7. Output Settings
        output_settings_frame = tk.LabelFrame(control_panel, text="7. Output Settings", padx=10, pady=10, bg="white")
        output_settings_frame.pack(pady=5, padx=10, fill=tk.X)
        self.save_bw_var = tk.BooleanVar(value=True)
        tk.Checkbutton(output_settings_frame, text="Save B&W Mask Image", variable=self.save_bw_var, bg="white").pack(anchor='w')

        # 8. Calculation (for single image mode)
        calc_frame = tk.LabelFrame(control_panel, text="8. Run Calculation (Single Image)", padx=10, pady=10, bg="white")
        calc_frame.pack(pady=5, padx=10, fill=tk.X)
        tk.Button(calc_frame, text="Calculate", command=self.run_calculation, bg="lightblue").pack(fill=tk.X, ipady=5)

        # 9. Results
        result_frame = tk.LabelFrame(control_panel, text="9. Result (Single Image)", padx=10, pady=10, bg="white")
        result_frame.pack(pady=5, padx=10, fill=tk.X)
        self.result_var = tk.StringVar(value="N/A")
        self.coverage_var = tk.StringVar(value="N/A")
        tk.Label(result_frame, textvariable=self.result_var, font=("Helvetica", 14, "bold"), bg="white").pack()
        tk.Label(result_frame, textvariable=self.coverage_var, font=("Helvetica", 10), bg="white").pack()

    def on_method_change(self, *args):
        """Enable/disable controls based on the selected method."""
        is_deterministic = "Largest Square" in self.method_var.get() or "Distance Transform" in self.method_var.get()
        
        if is_deterministic:
            self.lines_entry.config(state=tk.DISABLED)
            self.analyze_all_cb.config(state=tk.NORMAL)
        else:
            self.lines_entry.config(state=tk.NORMAL)
            self.analyze_all_cb.config(state=tk.DISABLED)
            self.analyze_all_var.set(False) # Uncheck when disabled

    def get_item_count(self):
        """Safely gets the number of lines/boxes from the entry box."""
        try:
            items = int(self.lines_var.get())
            if items <= 0:
                messagebox.showwarning("Invalid Input", "Item count must be a positive number. Defaulting to 5000.")
                return 5000
            return items
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for the item count. Defaulting to 5000.")
            return 5000

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.image_path = path
        try:
            self.original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if self.original_image is None: raise ValueError("Could not read image file.")
            self.canvas.delete(self.welcome_message)
            self.reset_polygon()
            self.reset_pixel_picker()
            self.update_image_processing()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path: return
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if not image_files:
            messagebox.showinfo("No Images Found", "The selected folder does not contain any supported image files.")
            return

        self.batch_results = []
        method = self.method_var.get()
        num_items = self.get_item_count() if "Largest Square" not in method and "Distance Transform" not in method else None
        
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Analyzing...")
        progress_win.geometry("300x100")
        tk.Label(progress_win, text="Processing images, please wait...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_win, orient=tk.HORIZONTAL, length=280, mode='determinate')
        progress_bar.pack(pady=10)
        progress_bar['maximum'] = len(image_files)

        for i, filename in enumerate(image_files):
            full_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue

                # Use inclusive thresholding
                binary_image = (img >= self.threshold_var.get()).astype(np.uint8) * 255
                if self.invert_var.get(): binary_image = cv2.bitwise_not(binary_image)
                
                full_binary_mask = (binary_image / 255).astype(np.uint8)
                labeled_array, num_features = label(full_binary_mask)
                if num_features == 0:
                    self.batch_results.append((filename, "N/A", "N/A"))
                    continue

                component_sizes = np.bincount(labeled_array.ravel())[1:]
                largest_component_label = component_sizes.argmax() + 1
                largest_component_mask = (labeled_array == largest_component_label).astype(np.uint8)
                
                # Decide which mask to analyze based on user setting
                if ("Largest Square" in method or "Distance Transform" in method) and self.analyze_all_var.get():
                    mask_to_analyze = full_binary_mask
                else:
                    mask_to_analyze = largest_component_mask

                if self.save_bw_var.get():
                    base, ext = os.path.splitext(full_path)
                    bw_path = f"{base}_bw{ext}"
                    # Save the correct mask based on user setting
                    mask_to_save = full_binary_mask if self.show_all_voids_var.get() else largest_component_mask
                    cv2.imwrite(bw_path, mask_to_save * 255)

                if method == "Mean Free Path (Line Intercept)":
                    result_pixels = self.calculate_mean_free_path(mask_to_analyze, num_lines=num_items)
                elif method == "Free Space (Box Counting)":
                    result_pixels = self.calculate_free_space_length(mask_to_analyze, num_squares=num_items)
                elif method == "Free Space (Distance Transform)":
                    result_pixels = self.calculate_free_space_distance_transform(mask_to_analyze)
                else: # Largest Square
                    result_pixels = self.calculate_free_space_largest_square(mask_to_analyze)

                real_world_width = float(self.width_var.get())
                scale = real_world_width / img.shape[1]
                result_microns = result_pixels * scale

                # Apply the cap specifically for the Distance Transform method
                if method == "Free Space (Distance Transform)" and result_microns > real_world_width:
                    result_microns = real_world_width
                
                # --- CORRECTED COVERAGE FRACTION LOGIC FOR BATCH ---
                material_pixels = np.sum(binary_image == 0)
                coverage_fraction = material_pixels / img.size
                
                self.batch_results.append((filename, f"{result_microns:.2f}", f"{coverage_fraction:.4f}"))

            except Exception as e:
                self.batch_results.append((filename, f"Error: {e}", "Error"))
            
            progress_bar['value'] = i + 1
            progress_win.update_idletasks()

        progress_win.destroy()
        self.show_results_window()
        
    def show_results_window(self):
        results_win = tk.Toplevel(self.root)
        results_win.title("Batch Analysis Results")
        results_win.geometry("750x400")
        tree_frame = tk.Frame(results_win)
        tree_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(tree_frame, columns=("Filename", "Result", "Coverage"), show="headings")
        tree.heading("Filename", text="Filename")
        tree.heading("Result", text=f"{self.method_var.get()} (µm)")
        tree.heading("Coverage", text="Coverage Fraction")
        tree.column("Filename", width=350)
        tree.column("Result", width=200, anchor=tk.CENTER)
        tree.column("Coverage", width=150, anchor=tk.CENTER)
        for item in self.batch_results:
            tree.insert("", tk.END, values=item)
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        button_frame = tk.Frame(results_win)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Copy to Clipboard", command=self.copy_results).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Export to CSV", command=self.export_results).pack(side=tk.LEFT, padx=10)

    def copy_results(self):
        if not self.batch_results: return
        header = f"Filename\t{self.method_var.get()} (µm)\tCoverage Fraction\n"
        content = header + "\n".join([f"{row[0]}\t{row[1]}\t{row[2]}" for row in self.batch_results])
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        messagebox.showinfo("Copied", "Results have been copied to the clipboard.")

    def export_results(self):
        if not self.batch_results: return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not filepath: return
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Filename", f"{self.method_var.get()} (µm)", "Coverage Fraction"])
                writer.writerows(self.batch_results)
            messagebox.showinfo("Export Successful", f"Results saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save file: {e}")

    def update_threshold_display(self, value):
        threshold = int(float(value))
        self.threshold_label.config(text=f"Value: {threshold}")
        if self.original_image is not None:
            self.update_image_processing(threshold_value=threshold)

    def update_image_processing(self, threshold_value=None):
        if self.original_image is None: return
        
        current_threshold = threshold_value if threshold_value is not None else self.threshold_var.get()

        # Use inclusive thresholding
        binary_image = (self.original_image >= current_threshold).astype(np.uint8) * 255
        if self.invert_var.get(): binary_image = cv2.bitwise_not(binary_image)
        
        self.full_binary_mask = (binary_image / 255).astype(np.uint8)
        
        labeled_array, num_features = label(self.full_binary_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled_array.ravel())[1:]
            largest_component_label = component_sizes.argmax() + 1
            self.largest_component_mask = (labeled_array == largest_component_label).astype(np.uint8)
        else:
            self.largest_component_mask = np.zeros_like(self.full_binary_mask)
            
        if self.save_bw_var.get() and self.image_path:
            base, ext = os.path.splitext(self.image_path)
            bw_path = f"{base}_bw{ext}"
            # Save the correct mask based on user setting
            mask_to_save = self.full_binary_mask if self.show_all_voids_var.get() else self.largest_component_mask
            cv2.imwrite(bw_path, mask_to_save * 255)
            
        display_image_bgr = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        
        # Decide which mask to show for the overlay
        if self.show_all_voids_var.get():
            mask_for_display = self.full_binary_mask
        else:
            mask_for_display = self.largest_component_mask

        overlay = np.zeros_like(display_image_bgr)
        overlay[mask_for_display == 1] = [255, 255, 0] # Cyan
        self.processed_image = cv2.addWeighted(display_image_bgr, 0.7, overlay, 0.3, 0)
        self.display_image()

    def display_image(self):
        if self.processed_image is None: return
        self.canvas.delete("all")
        img = Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            self.canvas.after(100, self.display_image)
            return
        img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(canvas_w / 2, canvas_h / 2, anchor=tk.CENTER, image=self.photo_image)
        self.draw_polygon()
        self.draw_picked_point()

    def toggle_polygon_drawing(self):
        if self.is_picking_pixel: self.toggle_pixel_picking()
        self.is_drawing_polygon = not self.is_drawing_polygon
        if self.is_drawing_polygon:
            self.poly_button.config(text="Finish Sampling", relief=tk.SUNKEN)
            self.canvas.config(cursor="crosshair")
        else:
            self.poly_button.config(text="Start Sampling", relief=tk.RAISED)
            self.canvas.config(cursor="")
        self.draw_polygon()

    def toggle_pixel_picking(self):
        if self.is_drawing_polygon: self.toggle_polygon_drawing()
        self.is_picking_pixel = not self.is_picking_pixel
        if self.is_picking_pixel:
            self.pixel_pick_button.config(text="Picking...", relief=tk.SUNKEN)
            self.canvas.config(cursor="crosshair")
        else:
            self.pixel_pick_button.config(text="Pick Pixel", relief=tk.RAISED)
            self.canvas.config(cursor="")

    def reset_polygon(self):
        self.polygon_points = []
        if self.is_drawing_polygon: self.toggle_polygon_drawing()
        if hasattr(self, 'photo_image'): self.display_image()
    
    def reset_pixel_picker(self):
        self.picked_point_canvas_coords = None
        self.pixel_value_var.set("Value: N/A")
        if self.is_picking_pixel: self.toggle_pixel_picking()
        if hasattr(self, 'photo_image'): self.display_image()

    def on_canvas_click(self, event):
        if self.is_drawing_polygon:
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            img_w, img_h = self.photo_image.width(), self.photo_image.height()
            offset_x = (canvas_w - img_w) / 2
            offset_y = (canvas_h - img_h) / 2
            if offset_x <= event.x < offset_x + img_w and offset_y <= event.y < offset_y + img_h:
                self.polygon_points.append((event.x, event.y))
                self.draw_polygon()
        elif self.is_picking_pixel:
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            img_w, img_h = self.photo_image.width(), self.photo_image.height()
            orig_w, orig_h = self.original_image.shape[1], self.original_image.shape[0]
            scale_x, scale_y = orig_w / img_w, orig_h / img_h
            offset_x, offset_y = (canvas_w - img_w) / 2, (canvas_h - img_h) / 2
            
            if offset_x <= event.x < offset_x + img_w and offset_y <= event.y < offset_y + img_h:
                img_coord_x = int((event.x - offset_x) * scale_x)
                img_coord_y = int((event.y - offset_y) * scale_y)
                
                pixel_value = self.original_image[img_coord_y, img_coord_x]
                self.pixel_value_var.set(f"Value: {pixel_value}")
                self.picked_point_canvas_coords = (event.x, event.y)
                self.draw_picked_point()
                self.toggle_pixel_picking() # Automatically exit picking mode

    def draw_polygon(self):
        self.canvas.delete("polygon")
        if len(self.polygon_points) > 1:
            self.canvas.create_line(self.polygon_points, fill="magenta", width=2, tags="polygon")
        if not self.is_drawing_polygon and len(self.polygon_points) > 2:
            self.canvas.create_line(self.polygon_points[-1], self.polygon_points[0], fill="magenta", width=2, tags="polygon")
        for p in self.polygon_points:
            x1, y1, x2, y2 = (p[0] - 3), (p[1] - 3), (p[0] + 3), (p[1] + 3)
            self.canvas.create_oval(x1, y1, x2, y2, fill="magenta", outline="black", tags="polygon")

    def draw_picked_point(self):
        self.canvas.delete("pixel_point")
        if self.picked_point_canvas_coords:
            p = self.picked_point_canvas_coords
            x1, y1, x2, y2 = (p[0] - 4), (p[1] - 4), (p[0] + 4), (p[1] + 4)
            self.canvas.create_oval(x1, y1, x2, y2, fill="red", outline="white", tags="pixel_point")

    def run_calculation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load a single image first.")
            return
        self.result_var.set("Calculating...")
        self.coverage_var.set("")
        self.root.update_idletasks()

        try:
            # --- NEW POLYGON LOGIC ---
            # If a polygon exists, use it to set the threshold
            if len(self.polygon_points) > 2:
                # 1. Create a mask from the polygon in the original image's coordinates
                canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
                img_w, img_h = self.photo_image.width(), self.photo_image.height()
                orig_w, orig_h = self.original_image.shape[1], self.original_image.shape[0]
                scale_x, scale_y = orig_w / img_w, orig_h / img_h
                offset_x, offset_y = (canvas_w - img_w) / 2, (canvas_h - img_h) / 2
                
                scaled_polygon = []
                for p in self.polygon_points:
                    img_coord_x = (p[0] - offset_x) * scale_x
                    img_coord_y = (p[1] - offset_y) * scale_y
                    scaled_polygon.append([img_coord_x, img_coord_y])
                
                poly_mask = np.zeros_like(self.original_image)
                cv2.fillPoly(poly_mask, [np.array(scaled_polygon, dtype=np.int32)], 255)

                # 2. Get all pixel values from the original image that are inside the polygon
                pixels_in_poly = self.original_image[poly_mask == 255]

                # 3. Find the most frequent pixel value (mode) and set it as the new threshold
                if pixels_in_poly.size > 0:
                    values, counts = np.unique(pixels_in_poly, return_counts=True)
                    mode = values[np.argmax(counts)]
                    std_dev = np.std(pixels_in_poly)
                    
                    # Set the new threshold as mode - std_dev for robustness
                    new_threshold = int(mode - std_dev)
                    
                    # Clamp the value to the valid 0-255 range
                    new_threshold = max(0, min(new_threshold, 255))
                    
                    self.threshold_var.set(new_threshold)
                    # Explicitly pass the new threshold to ensure the correct mask is generated
                    self.update_image_processing(threshold_value=new_threshold) 
                    self.reset_polygon() # Clear the polygon for the next operation
                else:
                    messagebox.showwarning("Warning", "Could not sample pixels from the drawn polygon.")

            # --- Proceed with calculation using the current mask ---
            # Decide which mask to analyze based on user setting
            if ("Largest Square" in self.method_var.get() or "Distance Transform" in self.method_var.get()) and self.analyze_all_var.get():
                mask_to_analyze = self.full_binary_mask
            else:
                mask_to_analyze = self.largest_component_mask

            if mask_to_analyze is None:
                messagebox.showerror("Error", "No valid area to analyze. Adjust threshold.")
                self.result_var.set("Error")
                return

            method = self.method_var.get()

            if method == "Mean Free Path (Line Intercept)":
                num_items = self.get_item_count()
                result_pixels = self.calculate_mean_free_path(mask_to_analyze, num_lines=num_items)
            elif method == "Free Space (Box Counting)":
                num_items = self.get_item_count()
                result_pixels = self.calculate_free_space_length(mask_to_analyze, num_squares=num_items)
            elif method == "Free Space (Distance Transform)":
                result_pixels = self.calculate_free_space_distance_transform(mask_to_analyze)
            else: # Largest Square
                result_pixels = self.calculate_free_space_largest_square(mask_to_analyze)

            real_world_width = float(self.width_var.get())
            scale = real_world_width / self.original_image.shape[1]
            result_microns = result_pixels * scale

            # Apply the cap specifically for the Distance Transform method
            if method == "Free Space (Distance Transform)" and result_microns > real_world_width:
                result_microns = real_world_width

            self.result_var.set(f"{result_microns:.2f} µm")

            # --- UNIFIED COVERAGE FRACTION LOGIC ---
            # Create the binary image based on the current threshold and invert settings.
            binary_image = (self.original_image >= self.threshold_var.get()).astype(np.uint8) * 255
            if self.invert_var.get():
                binary_image = cv2.bitwise_not(binary_image)
            
            # In the final binary_image, the void is white (255) and the material is black (0).
            # Therefore, counting the black pixels gives the material coverage.
            material_pixels = np.sum(binary_image == 0)
            coverage_fraction = material_pixels / self.original_image.size
            
            self.coverage_var.set(f"Coverage Fraction: {coverage_fraction:.4f}")

        except Exception as e:
            self.result_var.set("Error")
            self.coverage_var.set("")
            messagebox.showerror("Calculation Error", str(e))

    def calculate_mean_free_path(self, mask, num_lines=50000):
        h, w = mask.shape
        total_length, intercept_count = 0, 0
        x1, y1 = np.random.uniform(0, w, num_lines), np.random.uniform(0, h, num_lines)
        x2, y2 = np.random.uniform(0, w, num_lines), np.random.uniform(0, h, num_lines)
        for i in range(num_lines):
            length = int(np.hypot(x2[i]-x1[i], y2[i]-y1[i]))
            if length == 0: continue
            x = np.linspace(x1[i], x2[i], length).astype(int)
            y = np.linspace(y1[i], y2[i], length).astype(int)
            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = x[valid], y[valid]
            if len(x) == 0: continue
            line_values = mask[y, x]
            if np.sum(line_values) > 0:
                total_length += np.sum(line_values)
                diffs = np.diff(line_values)
                intercepts = np.sum(diffs != 0)
                if line_values[0] == 1: intercept_count += 1
                intercept_count += intercepts
        return total_length / intercept_count if intercept_count > 0 else 0

    def calculate_free_space_length(self, mask, num_squares=5000):
        h, w = mask.shape
        material_mask = 1 - mask 
        
        box_size = w // 4  
        
        for _ in range(10): 
            if box_size > min(h, w) or box_size < 1:
                box_size = max(1, min(h, w, box_size))

            if w - box_size <= 0 or h - box_size <= 0:
                if box_size > 1:
                    box_size = int(box_size * 0.75)
                    continue
                else:
                    return 1

            rand_x = np.random.randint(0, w - box_size, num_squares)
            rand_y = np.random.randint(0, h - box_size, num_squares)
            
            counts = []
            for i in range(num_squares):
                sub_matrix = material_mask[rand_y[i]:rand_y[i]+box_size, rand_x[i]:rand_x[i]+box_size]
                counts.append(np.sum(sub_matrix))
            
            values, freqs = np.unique(counts, return_counts=True)
            mode_index = np.argmax(freqs)
            
            if values[mode_index] == 0:
                original_mode_freq = freqs[mode_index]
                freqs[mode_index] = 0 
                if freqs.size > 1:
                    second_most_freq = np.max(freqs)
                    if second_most_freq / original_mode_freq < 0.05:
                        return box_size
                else: # Only one frequency (all zeros)
                    return box_size

                box_size = int(box_size * 1.2)
            else: 
                box_size = int(box_size * 0.75)

        return box_size 

    def calculate_free_space_distance_transform(self, mask):
        """
        Calculates the free space length using the Distance Transform (based on largest inscribed circle).
        """
        # Handle all-white mask case
        if np.all(mask == 1):
            return min(mask.shape) * np.sqrt(2)

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_radius = np.max(dist_transform)
        side_length = max_radius * np.sqrt(2)
        return side_length

    def calculate_free_space_largest_square(self, mask):
        """
        Calculates the side length of the largest square that can be inscribed
        in the void space using a dynamic programming approach.
        """
        if np.all(mask == 0):
            return 0
            
        # The input mask has 1s for void and 0s for material. This is the correct format.
        dp = mask.copy().astype(np.int32)
        h, w = dp.shape
        
        for i in range(1, h):
            for j in range(1, w):
                if dp[i, j] == 1:
                    dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
        
        # The max value in the dp table is the side length of the largest square
        max_size = np.max(dp)
        return max_size

if __name__ == '__main__':
    root = tk.Tk()
    app = MeanPathApp(root)
    root.mainloop()

