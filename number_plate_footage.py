import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
import os
import difflib
from datetime import datetime
import random
import csv
import webcolors
from colorthief import ColorThief
from io import BytesIO
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import pandas as pd

# Function to compute similarity between two strings
def is_similar(str1, str2, threshold=0.6):
    ratio = difflib.SequenceMatcher(None, str1, str2).ratio()
    return ratio >= threshold, ratio

# Updated function to get the closest color name from RGB
def get_color_name(rgb_color):
    min_distance = float('inf')
    closest_color = None
    # Using webcolors.names("css3") to get CSS3 color names
    for name in webcolors.names("css3"):
        try:
            color_rgb = webcolors.name_to_rgb(name)
            distance = np.linalg.norm(np.array(color_rgb) - np.array(rgb_color))
            if distance < min_distance:
                min_distance = distance
                closest_color = name
        except ValueError:
            continue
    return closest_color.capitalize() if closest_color else "Unknown"

# Function to estimate velocity with smoothing
def estimate_velocity(prev_positions, curr_center, time_elapsed, pixels_to_meters):
    if len(prev_positions) == 0 or time_elapsed <= 0:
        return 0

    # Append current position
    prev_positions.append(curr_center)

    # Keep only the last N positions
    N = 5  # Smoothing over last N positions
    if len(prev_positions) > N:
        prev_positions.pop(0)

    # Calculate average movement
    distances = []
    for i in range(len(prev_positions) - 1):
        dist = np.linalg.norm(np.array(prev_positions[i+1]) - np.array(prev_positions[i]))
        distances.append(dist)
    if len(distances) == 0:
        return 0
    avg_distance_pixels = sum(distances) / len(distances)

    # Convert pixels to meters
    avg_distance_meters = avg_distance_pixels * pixels_to_meters

    # Speed in meters per second
    speed_mps = avg_distance_meters / (time_elapsed / len(distances))

    # Apply logarithmic scaling to dampen large speed values
    speed_mps = np.log1p(speed_mps) * 10  # Adjust scaling factor as needed

    # Convert to km/h
    speed_kmh = speed_mps * 3.6

    return speed_kmh

# Function to get vehicle type from class ID
def get_vehicle_type(class_id):
    vehicle_classes = {
        2: 'Car',
        3: 'Motorcycle',
        5: 'Bus',
        7: 'Truck',
        # Add more classes as needed
    }
    return vehicle_classes.get(class_id, 'Other')

# Function to process video
def process_video(video_path, target_plate, pixels_to_meters, use_existing_csv):
    # Check if CSV exists
    csv_file = 'vehicles_data.csv'
    if use_existing_csv and os.path.exists(csv_file):
        messagebox.showinfo("Info", "Using existing CSV data.")
        return

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)

    # Load the YOLO models
    license_plate_model = YOLO('license_plate_detector.pt')
    vehicle_model = YOLO('yolov5s.pt')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        messagebox.showerror("Error", "Error opening video file")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps

    # Prepare to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'vehicles_detected.mp4'
    out = None

    # Prepare data storage
    vehicles_data = {}
    prev_vehicle_positions = {}
    vehicle_ids = 0

    frame_number = 0

    # Get the target number plate
    search_target = bool(target_plate)

    # Dictionary to store previous vehicle information
    prev_vehicles = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        timestamp = frame_number / fps  # Calculate timestamp in seconds

        # Perform vehicle detection
        vehicle_results = vehicle_model(frame)

        # Extract vehicle detections
        vehicle_detections = vehicle_results[0].boxes.xyxy.cpu().numpy()
        vehicle_confidences = vehicle_results[0].boxes.conf.cpu().numpy()
        vehicle_class_ids = vehicle_results[0].boxes.cls.cpu().numpy()

        current_vehicles = []

        for i, vehicle_box in enumerate(vehicle_detections):
            vx_min, vy_min, vx_max, vy_max = map(int, vehicle_box)
            vehicle_confidence = vehicle_confidences[i]
            vehicle_class_id = int(vehicle_class_ids[i])

            # Filter out low-confidence detections
            if vehicle_confidence < 0.5:
                continue

            # Filter detections to only vehicle classes
            if vehicle_class_id not in [2, 3, 5, 7]:  # Vehicle class IDs
                continue

            # Get vehicle ROI
            vehicle_roi = frame[vy_min:vy_max, vx_min:vx_max]

            # Estimate vehicle color
            # Convert ROI to PIL Image for ColorThief
            _, buffer = cv2.imencode('.png', vehicle_roi)
            color_thief = ColorThief(BytesIO(buffer))
            dominant_color = color_thief.get_color(quality=1)
            color_name = get_color_name(dominant_color)

            # Estimate vehicle type
            vehicle_type = get_vehicle_type(vehicle_class_id)

            # Current center of the vehicle
            curr_center = ((vx_min + vx_max) // 2, (vy_min + vy_max) // 2)

            # Attempt to match with existing vehicles
            matched_id = None
            min_distance = float('inf')
            for vid, pdata in prev_vehicles.items():
                prev_center = pdata['center']
                distance = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
                if distance < 50 and distance < min_distance:
                    min_distance = distance
                    matched_id = vid

            if matched_id is None:
                # Assign a new ID to the vehicle
                vehicle_ids += 1
                matched_id = vehicle_ids
                vehicles_data[matched_id] = {
                    'Vehicle ID': matched_id,
                    'Number Plate': "Unknown",
                    'Vehicle Type': vehicle_type,
                    'Color': color_name,
                    'Velocity (km/h)': 0,
                    'Latitude': round(random.uniform(-90.0, 90.0), 6),
                    'Longitude': round(random.uniform(-180.0, 180.0), 6),
                    'Start Time (s)': round(timestamp, 2),
                    'End Time (s)': round(timestamp, 2),
                }
                prev_vehicle_positions[matched_id] = []

            # Estimate velocity
            prev_data = prev_vehicles.get(matched_id, {})
            time_elapsed = timestamp - prev_data.get('timestamp', timestamp)
            prev_positions = prev_vehicle_positions.get(matched_id, [])
            velocity = estimate_velocity(prev_positions, curr_center, time_elapsed if time_elapsed > 0 else frame_duration, pixels_to_meters)

            # Update vehicle data
            prev_vehicles[matched_id] = {
                'center': curr_center,
                'timestamp': timestamp
            }
            prev_vehicle_positions[matched_id] = prev_positions

            # Update vehicle info
            vehicles_data[matched_id]['Velocity (km/h)'] = round(velocity, 2)
            vehicles_data[matched_id]['End Time (s)'] = round(timestamp, 2)
            vehicles_data[matched_id]['Color'] = color_name  # Update color in case it changes

            # Perform license plate detection on vehicle ROI
            license_plate_results = license_plate_model(vehicle_roi)

            # Extract license plate detections
            lp_detections = license_plate_results[0].boxes.xyxy.cpu().numpy()
            lp_confidences = license_plate_results[0].boxes.conf.cpu().numpy()
            lp_class_ids = license_plate_results[0].boxes.cls.cpu().numpy()

            number_plate_text = vehicles_data[matched_id]['Number Plate']

            for j, lp_box in enumerate(lp_detections):
                x_min, y_min, x_max, y_max = map(int, lp_box)
                lp_confidence = lp_confidences[j]
                lp_class_id = int(lp_class_ids[j])

                # Filter out low-confidence detections
                if lp_confidence < 0.5:
                    continue

                # Crop the license plate from the vehicle ROI
                plate_image = vehicle_roi[y_min:y_max, x_min:x_max]

                # Preprocess the plate image for OCR
                # Convert to grayscale
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

                # Increase contrast
                alpha = 1.5  # Contrast control
                beta = 0     # Brightness control
                adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

                # Apply Gaussian Blur
                blurred = cv2.GaussianBlur(adjusted, (3, 3), 0)

                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 31, 2)

                # OCR using EasyOCR
                ocr_result = reader.readtext(thresh)

                if ocr_result:
                    # Assuming the largest text area is the plate
                    ocr_result = sorted(ocr_result, key=lambda x: x[1], reverse=True)
                    number_plate_text = ocr_result[0][-2]
                    # Clean the text
                    number_plate_text = ''.join(e for e in number_plate_text if e.isalnum()).upper()
                    vehicles_data[matched_id]['Number Plate'] = number_plate_text
                    break  # Assuming one plate per vehicle

            # Check if we need to match with the target plate
            match_found = False
            if search_target:
                similar, similarity_ratio = is_similar(number_plate_text, target_plate)
                match_found = similar
                if similar:
                    print(f"Matched with target plate '{target_plate}' (Similarity: {similarity_ratio:.2f})")
            else:
                match_found = True  # Process all vehicles if no target plate is provided

            # Save frames for video stitching
            if match_found:
                # Draw bounding boxes and annotations
                cv2.rectangle(frame, (vx_min, vy_min), (vx_max, vy_max), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {matched_id}", (vx_min, vy_min - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"{vehicle_type}", (vx_min, vy_min - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"{number_plate_text}", (vx_min, vy_min - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Speed: {round(velocity, 2)} km/h", (vx_min, vy_min + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Color: {color_name}", (vx_min, vy_min + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 128, 255), 2)

                # Initialize video writer if not already done
                if out is None:
                    height, width = frame.shape[:2]
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                # Write the frame to the output video
                out.write(frame)

            # Keep track of current vehicles
            current_vehicles.append(matched_id)

        # Remove vehicles not detected in current frame
        vehicles_to_remove = [vid for vid in prev_vehicles if vid not in current_vehicles]
        for vid in vehicles_to_remove:
            del prev_vehicles[vid]
            del prev_vehicle_positions[vid]

    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    # Save data to a CSV file
    csv_file = 'vehicles_data.csv'
    with open(csv_file, mode='w', newline='') as file:
        fieldnames = ['Vehicle ID', 'Number Plate', 'Vehicle Type', 'Color', 'Velocity (km/h)', 'Latitude', 'Longitude', 'Start Time (s)', 'End Time (s)']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for data in vehicles_data.values():
            writer.writerow(data)

    messagebox.showinfo("Processing Complete", f"Vehicle data saved to '{csv_file}'.")
    if out is not None:
        messagebox.showinfo("Video Saved", f"Output video saved as '{output_video_path}'.")
    else:
        messagebox.showinfo("No Matching Vehicles", "No matching vehicles found.")

# Function to generate filtered video
def generate_filtered_video(filters):
    csv_file = 'vehicles_data.csv'
    if not os.path.exists(csv_file):
        messagebox.showerror("Error", "CSV file not found. Please process the video first.")
        return

    # Read CSV data
    df = pd.read_csv(csv_file)

    # Apply filters
    if filters['Vehicle ID']:
        df = df[df['Vehicle ID'] == int(filters['Vehicle ID'])]
    if filters['Number Plate']:
        df = df[df['Number Plate'].str.contains(filters['Number Plate'], na=False)]
    if filters['Vehicle Type'] != 'All':
        df = df[df['Vehicle Type'] == filters['Vehicle Type']]
    if filters['Color'] != 'All':
        df = df[df['Color'] == filters['Color']]
    if filters['Velocity']:
        df = df[(df['Velocity (km/h)'] >= filters['Velocity'][0]) & (df['Velocity (km/h)'] <= filters['Velocity'][1])]
    if filters['Timestamp']:
        df = df[(df['Start Time (s)'] >= filters['Timestamp'][0]) & (df['End Time (s)'] <= filters['Timestamp'][1])]

    if df.empty:
        messagebox.showinfo("No Results", "No vehicles match the selected filters.")
        return

    # Open the video file
    video_path = video_path_entry.get()
    cap = cv2.VideoCapture(video_path)

    # Prepare to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'filtered_vehicles.mp4'
    out = None

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Convert timestamps to frame numbers
    df['Start Frame'] = (df['Start Time (s)'] * fps).astype(int)
    df['End Frame'] = (df['End Time (s)'] * fps).astype(int)

    # Get unique frames to extract
    frames_to_extract = []
    for index, row in df.iterrows():
        frames_to_extract.extend(range(row['Start Frame'], row['End Frame'] + 1))
    frames_to_extract = sorted(set(frames_to_extract))

    # Read frames and write to output video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if frame_number in frames_to_extract:
            # Initialize video writer if not already done
            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Write frame to output video
            out.write(frame)

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    messagebox.showinfo("Filtered Video Saved", f"Filtered video saved as '{output_video_path}'.")

# GUI Implementation
def run_program():
    video_path = video_path_entry.get()
    target_plate = target_plate_entry.get().upper()
    use_existing_csv = process_option.get() == "Use Existing CSV"
    pixels_to_meters = float(scale_entry.get())

    if not os.path.exists(video_path):
        messagebox.showerror("Error", "Video file not found.")
        return

    # Run processing in a separate thread to keep GUI responsive
    threading.Thread(target=process_video, args=(video_path, target_plate, pixels_to_meters, use_existing_csv)).start()

def browse_video():
    filename = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    video_path_entry.delete(0, tk.END)
    video_path_entry.insert(0, filename)

def load_filters():
    csv_file = 'vehicles_data.csv'
    if not os.path.exists(csv_file):
        messagebox.showerror("Error", "CSV file not found. Please process the video first.")
        return

    df = pd.read_csv(csv_file)
    # Populate vehicle type dropdown
    vehicle_types = ['All'] + sorted(df['Vehicle Type'].dropna().unique().tolist())
    vehicle_type_dropdown['values'] = vehicle_types
    vehicle_type_dropdown.current(0)

    # Populate color dropdown
    colors = ['All'] + sorted(df['Color'].dropna().unique().tolist())
    color_dropdown['values'] = colors
    color_dropdown.current(0)

    # Set velocity slider range
    velocity_min = int(df['Velocity (km/h)'].min())
    velocity_max = int(df['Velocity (km/h)'].max())
    velocity_min_slider.config(from_=velocity_min, to=velocity_max)
    velocity_max_slider.config(from_=velocity_min, to=velocity_max)
    velocity_min_var.set(velocity_min)
    velocity_max_var.set(velocity_max)
    update_velocity_label()

    # Set timestamp slider range
    time_min = int(df['Start Time (s)'].min())
    time_max = int(df['End Time (s)'].max())
    timestamp_min_slider.config(from_=time_min, to=time_max)
    timestamp_max_slider.config(from_=time_min, to=time_max)
    timestamp_min_var.set(time_min)
    timestamp_max_var.set(time_max)
    update_timestamp_label()

def update_velocity_label(val=None):
    min_val = int(velocity_min_var.get())
    max_val = int(velocity_max_var.get())
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    velocity_value_label.config(text=f"{min_val} - {max_val} km/h")

def update_timestamp_label(val=None):
    min_val = int(timestamp_min_var.get())
    max_val = int(timestamp_max_var.get())
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    timestamp_value_label.config(text=f"{min_val} - {max_val} s")

def apply_filters():
    min_velocity = min(velocity_min_var.get(), velocity_max_var.get())
    max_velocity = max(velocity_min_var.get(), velocity_max_var.get())
    min_timestamp = min(timestamp_min_var.get(), timestamp_max_var.get())
    max_timestamp = max(timestamp_min_var.get(), timestamp_max_var.get())
    filters = {
        'Vehicle ID': vehicle_id_entry.get(),
        'Number Plate': number_plate_entry.get().upper(),
        'Vehicle Type': vehicle_type_var.get(),
        'Color': color_var.get(),
        'Velocity': (min_velocity, max_velocity),
        'Timestamp': (min_timestamp, max_timestamp),
    }
    threading.Thread(target=generate_filtered_video, args=(filters,)).start()

# Create the main window
root = tk.Tk()
root.title("Vehicle Detection System")

# Video Path
tk.Label(root, text="Video Path:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
video_path_entry = tk.Entry(root, width=50)
video_path_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=browse_video).grid(row=0, column=2, padx=5, pady=5)

# Target Number Plate
tk.Label(root, text="Target Number Plate (optional):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
target_plate_entry = tk.Entry(root, width=20)
target_plate_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

# Scale Entry
tk.Label(root, text="Pixels to Meters Scale Factor:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
scale_entry = tk.Entry(root, width=10)
scale_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
scale_entry.insert(0, "0.05")  # Default value, adjust as needed

# Processing Options
process_option = tk.StringVar(value="Process Video")
tk.Radiobutton(root, text="Process Video", variable=process_option, value="Process Video").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
tk.Radiobutton(root, text="Use Existing CSV", variable=process_option, value="Use Existing CSV").grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

# Buttons
tk.Button(root, text="Run", command=run_program).grid(row=4, column=0, padx=5, pady=10)
tk.Button(root, text="Load Filters", command=load_filters).grid(row=4, column=1, padx=5, pady=10)
tk.Button(root, text="Exit", command=root.quit).grid(row=4, column=2, padx=5, pady=10)

# Separator
separator = ttk.Separator(root, orient='horizontal')
separator.grid(row=5, column=0, columnspan=3, sticky='ew', padx=5, pady=10)

# Filtering Options
tk.Label(root, text="Filter Options:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)

# Vehicle ID
tk.Label(root, text="Vehicle ID:").grid(row=7, column=0, padx=5, pady=5, sticky=tk.E)
vehicle_id_entry = tk.Entry(root, width=10)
vehicle_id_entry.grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)

# Number Plate
tk.Label(root, text="Number Plate:").grid(row=8, column=0, padx=5, pady=5, sticky=tk.E)
number_plate_entry = tk.Entry(root, width=20)
number_plate_entry.grid(row=8, column=1, padx=5, pady=5, sticky=tk.W)

# Vehicle Type
tk.Label(root, text="Vehicle Type:").grid(row=9, column=0, padx=5, pady=5, sticky=tk.E)
vehicle_type_var = tk.StringVar(value='All')
vehicle_type_dropdown = ttk.Combobox(root, textvariable=vehicle_type_var, state='readonly')
vehicle_type_dropdown.grid(row=9, column=1, padx=5, pady=5, sticky=tk.W)

# Color
tk.Label(root, text="Color:").grid(row=10, column=0, padx=5, pady=5, sticky=tk.E)
color_var = tk.StringVar(value='All')
color_dropdown = ttk.Combobox(root, textvariable=color_var, state='readonly')
color_dropdown.grid(row=10, column=1, padx=5, pady=5, sticky=tk.W)

# Velocity Min Slider
tk.Label(root, text="Min Velocity (km/h):").grid(row=11, column=0, padx=5, pady=5, sticky=tk.E)
velocity_min_var = tk.DoubleVar()
velocity_min_slider = tk.Scale(root, from_=0, to=100, orient='horizontal', variable=velocity_min_var, command=update_velocity_label)
velocity_min_slider.grid(row=11, column=1, padx=5, pady=5, sticky=tk.W)

# Velocity Max Slider
tk.Label(root, text="Max Velocity (km/h):").grid(row=12, column=0, padx=5, pady=5, sticky=tk.E)
velocity_max_var = tk.DoubleVar()
velocity_max_slider = tk.Scale(root, from_=0, to=100, orient='horizontal', variable=velocity_max_var, command=update_velocity_label)
velocity_max_slider.grid(row=12, column=1, padx=5, pady=5, sticky=tk.W)

velocity_value_label = tk.Label(root, text="0 - 100 km/h")
velocity_value_label.grid(row=13, column=1, padx=5, pady=5, sticky=tk.W)

# Timestamp Min Slider
tk.Label(root, text="Min Timestamp (s):").grid(row=14, column=0, padx=5, pady=5, sticky=tk.E)
timestamp_min_var = tk.DoubleVar()
timestamp_min_slider = tk.Scale(root, from_=0, to=100, orient='horizontal', variable=timestamp_min_var, command=update_timestamp_label)
timestamp_min_slider.grid(row=14, column=1, padx=5, pady=5, sticky=tk.W)

# Timestamp Max Slider
tk.Label(root, text="Max Timestamp (s):").grid(row=15, column=0, padx=5, pady=5, sticky=tk.E)
timestamp_max_var = tk.DoubleVar()
timestamp_max_slider = tk.Scale(root, from_=0, to=100, orient='horizontal', variable=timestamp_max_var, command=update_timestamp_label)
timestamp_max_slider.grid(row=15, column=1, padx=5, pady=5, sticky=tk.W)

timestamp_value_label = tk.Label(root, text="0 - 100 s")
timestamp_value_label.grid(row=16, column=1, padx=5, pady=5, sticky=tk.W)

# Apply Filters Button
tk.Button(root, text="Apply Filters and Generate Video", command=apply_filters).grid(row=17, column=0, columnspan=2, padx=5, pady=10)

root.mainloop()
