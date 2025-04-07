import os
import re
import uuid
import io
import logging
import tempfile
import shutil
import time # For sleep in SSE stream
import json # For formatting SSE data
import threading # For background tasks
from datetime import datetime, timezone

import cv2
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend REQUIRED for Flask/servers
import matplotlib.pyplot as plt
import numpy as np
from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for, Response, jsonify, # Added Response, jsonify
                   stream_with_context) # Added stream_with_context
from werkzeug.utils import secure_filename

# --- Configuration ---
# (Keep previous configurations: UPLOAD_FOLDER, STATIC_FOLDER, etc.)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, 'output_files')
ALLOWED_EXTENSIONS = {'txt'}
FRAME_WIDTH = 32
FRAME_HEIGHT = 24
VIDEO_FPS = 15
OUTPUT_VIDEO_CODEC = 'mp4v'
OUTPUT_IMAGE_DPI = 90
GPS_PLOT_FIGSIZE = (5, 3.5)
THERMAL_PLOT_FIGSIZE = (5, 4)
PADDING_COLOR = [255, 255, 255]

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = os.urandom(24)

# --- Ensure directories exist ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Task Management (In-Memory - NOT SUITABLE FOR PRODUCTION) ---
tasks = {}
tasks_lock = threading.Lock() # To safely access tasks dict from multiple threads
VIDEO_EXPIRY_MINUTES = 15  # Videos will be deleted after 15 minutes

def cleanup_old_videos():
    """Removes videos that are older than VIDEO_EXPIRY_MINUTES."""
    try:
        current_time = datetime.now(timezone.utc)
        with tasks_lock:
            # Get list of tasks to remove
            tasks_to_remove = []
            for task_id, task_info in tasks.items():
                if task_info.get('status') == 'complete':
                    created_time = task_info.get('created_time')
                    if created_time and (current_time - created_time).total_seconds() > VIDEO_EXPIRY_MINUTES * 60:
                        tasks_to_remove.append(task_id)
                        # Delete the video file
                        video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{task_id}_therm_gps_realtime.mp4")
                        if os.path.exists(video_path):
                            try:
                                os.remove(video_path)
                                logger.info(f"Deleted expired video: {video_path}")
                            except OSError as e:
                                logger.error(f"Error deleting video {video_path}: {e}")

            # Remove expired tasks
            for task_id in tasks_to_remove:
                del tasks[task_id]
                logger.info(f"Removed expired task: {task_id}")
    except Exception as e:
        logger.exception(f"Error in cleanup_old_videos: {e}")

def cleanup_output_directory():
    """Cleans up the output directory on server start."""
    try:
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            for filename in os.listdir(app.config['OUTPUT_FOLDER']):
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted file on startup: {file_path}")
                except OSError as e:
                    logger.error(f"Error deleting {file_path}: {e}")
    except Exception as e:
        logger.exception(f"Error in cleanup_output_directory: {e}")

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_task_progress(task_id, status, percent, message="", result=None):
    """Safely updates the status of a task."""
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id]['status'] = status
            tasks[task_id]['percent'] = percent
            tasks[task_id]['message'] = message
            tasks[task_id]['result'] = result
            if status == 'complete':
                tasks[task_id]['created_time'] = datetime.now(timezone.utc)
            logger.debug(f"Task {task_id} update: Status={status}, Percent={percent}, Msg={message}")
        else:
            logger.warning(f"Attempted to update non-existent task {task_id}")

# --- Parsing and Image Generation Functions ---
# (Keep parse_data_file, generate_padded_image, generate_thermal_image,
#  generate_gps_plot_image identical to the previous version)
# ... (Include the full code for these 4 functions here from the previous response) ...
def parse_data_file(filepath):
    """Parses the uploaded text file and converts timestamps to datetime objects."""
    data_entries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        entry_pattern = re.compile(
            r"Timestamp:\s*(?P<timestamp>.*?)\s*\n"
            r"\s*Location:\s*Lat:\s*(?P<lat>[-+]?\d*\.?\d+)\s*,\s*Lon:\s*(?P<lon>[-+]?\d*\.?\d+)\s*\n"
            r"\s*Temperature Range:\s*\[\s*(?P<min_temp_range>[-+]?\d*\.?\d*)\s*,\s*(?P<max_temp_range>[-+]?\d*\.?\d*)\s*\]\s*\n"
            r"\s*Frame Data:\s*(?P<frame_data>[\d,\s.-]+?)\s*(?=\n\s*Timestamp:|\Z)",
            re.DOTALL | re.IGNORECASE)
        for match in entry_pattern.finditer(content):
            entry_dict = match.groupdict()
            try:
                timestamp_str = entry_dict['timestamp'].strip()
                timestamp_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S %z')
                timestamp_dt_utc = timestamp_dt.astimezone(timezone.utc)
                lat = float(entry_dict['lat'])
                lon = float(entry_dict['lon'])
                min_temp_range = float(entry_dict['min_temp_range'])
                max_temp_range = float(entry_dict['max_temp_range'])
                frame_data_str = re.sub(r'[\s\n]+', '', entry_dict['frame_data']).strip(',')
                frame_values = [float(x) for x in frame_data_str.split(',') if x]
                
                if len(frame_values) != FRAME_WIDTH * FRAME_HEIGHT:
                    logger.warning(f"Incorrect frame data length ({len(frame_values)}) for timestamp {timestamp_str}. Skipping.")
                    continue
                
                # Normalize frame values to 0-255 range
                frame_min = min(frame_values)
                frame_max = max(frame_values)
                if frame_max > frame_min:
                    normalized_values = [(x - frame_min) * 255 / (frame_max - frame_min) for x in frame_values]
                else:
                    normalized_values = [0] * len(frame_values)  # All values are the same
                
                data_entries.append({
                    'timestamp_str': timestamp_str,
                    'timestamp_dt': timestamp_dt_utc,
                    'lat': lat,
                    'lon': lon,
                    'min_temp': min_temp_range,
                    'max_temp': max_temp_range,
                    'frame_data': normalized_values
                })
            except Exception as e:
                logger.error(f"Error processing entry near timestamp {entry_dict.get('timestamp', 'N/A')}: {e}")
                continue
        if not data_entries:
            logger.error("No valid data entries found.")
            return None
        data_entries.sort(key=lambda x: x['timestamp_dt'])
        logger.info(f"Parsed {len(data_entries)} entries.")
        return data_entries
    except Exception as e:
        logger.exception(f"Error during file parsing: {e}")
        return None

def generate_padded_image(fig, target_width, padding_color):
    """Generates image from figure, decodes, and pads horizontally."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=OUTPUT_IMAGE_DPI, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        buf.close()
        plt.close(fig)
        if img is None: raise ValueError("Failed to decode image")
        current_h, current_w, _ = img.shape; pad_total = target_width - current_w
        pad_left = pad_total // 2; pad_right = pad_total - pad_left
        if pad_total < 0:
            scale = target_width / current_w; new_h = int(current_h * scale)
            img_padded = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
        elif pad_total > 0: img_padded = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=padding_color)
        else: img_padded = img
        return img_padded
    except Exception as e: plt.close(fig); logger.exception(f"Error padding image: {e}"); raise e # Re-raise

def generate_thermal_image(frame_data, min_temp, max_temp, timestamp_str, target_width):
    """Generates a single padded thermal image frame as a NumPy array."""
    fig = None # Ensure fig exists for potential closing in except block
    try:
        # Convert normalized values (0-255) to temperature range
        frame_np = np.array(frame_data).reshape((FRAME_HEIGHT, FRAME_WIDTH))
        temp_range = max_temp - min_temp
        if temp_range > 0:
            frame_np = min_temp + (frame_np * temp_range / 255.0)
        else:
            frame_np = np.full_like(frame_np, min_temp)
            
        # Rotate CCW 90 degrees and flip horizontally
        frame_np = np.rot90(frame_np, k=1)  # k=1 for CCW rotation
        frame_np = np.fliplr(frame_np)  # Flip horizontally
            
        fig, ax = plt.subplots(figsize=THERMAL_PLOT_FIGSIZE, dpi=OUTPUT_IMAGE_DPI)
        im = ax.imshow(frame_np, cmap='inferno', vmin=min_temp, vmax=max_temp, interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"Thermal Frame\n{timestamp_str}", fontsize=10)
        img_padded = generate_padded_image(fig, target_width, PADDING_COLOR) # This now closes the fig
        return img_padded
    except Exception as e: logger.exception(f"Error generating thermal image: {e}"); return None

def generate_gps_plot_image(lons, lats, target_width, lon_min, lon_max, lat_min, lat_max):
    """Generates a single padded GPS plot image as a NumPy array."""
    if not lons: return None
    fig = None
    try:
        # Create figure with 1:1 aspect ratio
        fig, ax = plt.subplots(figsize=GPS_PLOT_FIGSIZE, dpi=OUTPUT_IMAGE_DPI)
        
        # Plot the data
        if len(lons) > 1: ax.plot(lons, lats, 'b-', markersize=2, linewidth=1, label='Path')
        ax.plot(lons[-1], lats[-1], 'ro', markersize=4, label='Current')
        
        # Set the plot ranges
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        # Force 1:1 aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Set labels and formatting
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.set_title("GPS Location", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.ticklabel_format(useOffset=False, style='plain')
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
        
        img_padded = generate_padded_image(fig, target_width, PADDING_COLOR) # This now closes the fig
        return img_padded
    except Exception as e: logger.exception(f"Error generating GPS plot image: {e}"); return None

# --- Video Generation Task (Runs in Background Thread) ---
def run_video_task(task_id, upload_path, video_filename):
    """The function that runs in a separate thread to create the video."""
    logger.info(f"Background task {task_id} started for {video_filename}")
    video_output_path = os.path.join(app.config['OUTPUT_FOLDER'], video_filename)
    data_entries = None
    video_writer = None

    try:
        # 1. Parse Data
        update_task_progress(task_id, "processing", 5, "Parsing data file...")
        data_entries = parse_data_file(upload_path)
        if not data_entries or len(data_entries) < 2:
            raise ValueError("Not enough valid data entries found for video creation.")
        update_task_progress(task_id, "processing", 10, "Data parsed successfully.")

        # Calculate initial GPS range from all data points
        all_lons = [entry['lon'] for entry in data_entries]
        all_lats = [entry['lat'] for entry in data_entries]
        lon_min, lon_max = min(all_lons), max(all_lons)
        lat_min, lat_max = min(all_lats), max(all_lats)
        
        # Add 10% padding
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        padding = 0.1
        
        # Handle case where all points are the same
        if lon_range == 0:
            lon_min -= 0.0001
            lon_max += 0.0001
        else:
            lon_min -= lon_range * padding
            lon_max += lon_range * padding
            
        if lat_range == 0:
            lat_min -= 0.0001
            lat_max += 0.0001
        else:
            lat_min -= lat_range * padding
            lat_max += lat_range * padding

        # 2. Determine Video Frame Size (generate sample frames)
        update_task_progress(task_id, "processing", 15, "Determining video dimensions...")
        sample_lons = [e['lon'] for e in data_entries]
        sample_lats = [e['lat'] for e in data_entries]
        # Use the generation functions which include padding logic helper
        try:
             # Generate thermal sample to find its width/height before padding
             sample_thermal_fig_only, ax_th_s = plt.subplots(figsize=THERMAL_PLOT_FIGSIZE, dpi=OUTPUT_IMAGE_DPI)
             ax_th_s.imshow(np.zeros((FRAME_HEIGHT, FRAME_WIDTH)), cmap='inferno')
             plt.colorbar(ax_th_s.images[0], ax=ax_th_s)
             ax_th_s.axis('off')
             buf_th_s = io.BytesIO()
             sample_thermal_fig_only.savefig(buf_th_s, format='png', dpi=OUTPUT_IMAGE_DPI, bbox_inches='tight', pad_inches=0.1)
             buf_th_s.seek(0)
             th_img_s = cv2.imdecode(np.frombuffer(buf_th_s.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
             buf_th_s.close()
             plt.close(sample_thermal_fig_only)
             if th_img_s is None:
                 raise ValueError("Failed decode thermal sample")
             thermal_h, thermal_w, _ = th_img_s.shape

             # Generate GPS sample
             sample_gps_fig_only, ax_gps_s = plt.subplots(figsize=GPS_PLOT_FIGSIZE, dpi=OUTPUT_IMAGE_DPI)
             ax_gps_s.plot(sample_lons, sample_lats)
             ax_gps_s.plot(sample_lons[-1], sample_lats[-1], 'ro')
             ax_gps_s.set_xlim(lon_min, lon_max)
             ax_gps_s.set_ylim(lat_min, lat_max)
             ax_gps_s.set_aspect('equal', adjustable='box')
             buf_gps_s = io.BytesIO()
             sample_gps_fig_only.savefig(buf_gps_s, format='png', dpi=OUTPUT_IMAGE_DPI, bbox_inches='tight', pad_inches=0.1)
             buf_gps_s.seek(0)
             gps_img_s = cv2.imdecode(np.frombuffer(buf_gps_s.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
             buf_gps_s.close()
             plt.close(sample_gps_fig_only)
             if gps_img_s is None:
                 raise ValueError("Failed decode gps sample")
             gps_h, gps_w, _ = gps_img_s.shape

        except Exception as e:
            logger.exception("Error generating sample frames for dimensions")
            raise ValueError(f"Failed to determine video dimensions: {e}")


        combined_w = max(thermal_w, gps_w)
        combined_h = thermal_h + gps_h
        frame_size = (combined_w, combined_h)
        logger.info(f"Task {task_id}: Video frame size set to {frame_size} (WxH)")
        update_task_progress(task_id, "processing", 20, f"Video dimensions set ({combined_w}x{combined_h}). Initializing writer...")

        # 3. Initialize Video Writer
        fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
        video_writer = cv2.VideoWriter(video_output_path, fourcc, VIDEO_FPS, frame_size)
        if not video_writer.isOpened():
            raise IOError(f"Could not open video writer for path: {video_output_path}")

        # 4. Generate Frames and Write Video
        total_unique_frames = len(data_entries)
        current_lons = []
        current_lats = []
        start_process_percent = 25 # Percentage where frame processing starts
        end_process_percent = 95 # Percentage where frame processing ends

        for i in range(total_unique_frames):
            # Calculate progress percentage within the processing range
            percent_done = start_process_percent + int(((i + 1) / total_unique_frames) * (end_process_percent - start_process_percent))
            update_task_progress(task_id, "processing", percent_done, f"Processing frame {i+1}/{total_unique_frames}")

            entry = data_entries[i]
            current_lons.append(entry['lon'])
            current_lats.append(entry['lat'])

            # Generate images (already includes padding logic)
            img_thermal_padded = generate_thermal_image(entry['frame_data'], entry['min_temp'], entry['max_temp'], entry['timestamp_str'], combined_w)
            img_gps_padded = generate_gps_plot_image(current_lons, current_lats, combined_w, lon_min, lon_max, lat_min, lat_max)

            if img_thermal_padded is None or img_gps_padded is None:
                logger.warning(f"Task {task_id}: Skipping frame {i+1} due to image generation error.")
                continue # Skip this frame if generation failed

            # Stack images (thermal on top)
            # Re-verify heights match determined heights before stacking
            target_th_h, target_gps_h = thermal_h, gps_h
            if img_thermal_padded.shape[0] != target_th_h: img_thermal_padded = cv2.resize(img_thermal_padded, (combined_w, target_th_h))
            if img_gps_padded.shape[0] != target_gps_h: img_gps_padded = cv2.resize(img_gps_padded, (combined_w, target_gps_h))

            combined_frame = np.vstack((img_thermal_padded, img_gps_padded))

            # Final resize check
            if combined_frame.shape[0] != combined_h or combined_frame.shape[1] != combined_w:
                combined_frame = cv2.resize(combined_frame, frame_size, interpolation=cv2.INTER_AREA)

            # Calculate duration and duplicates
            if i < total_unique_frames - 1:
                time_delta = data_entries[i+1]['timestamp_dt'] - entry['timestamp_dt']
                duration_sec = max(0, time_delta.total_seconds())
            else:
                duration_sec = 1.0 / VIDEO_FPS
            num_duplicates = max(1, round(duration_sec * VIDEO_FPS))

            # Write duplicates
            for _ in range(num_duplicates):
                video_writer.write(combined_frame)

        # 5. Finalize
        update_task_progress(task_id, "processing", 98, "Finalizing video file...")
        if video_writer: video_writer.release()

        # Generate relative path for download URL
        relative_video_path = os.path.join('output_files', video_filename)
        update_task_progress(task_id, "complete", 100, "Video generation complete!", result=relative_video_path)
        logger.info(f"Background task {task_id} completed successfully.")

    except Exception as e:
        logger.exception(f"Error in background task {task_id}: {e}")
        update_task_progress(task_id, "error", tasks.get(task_id, {}).get('percent', 0) , f"Error: {e}")
        # Clean up partially created video file on error
        if video_writer and video_writer.isOpened(): video_writer.release()
        if os.path.exists(video_output_path):
            try: os.remove(video_output_path)
            except OSError: logger.warning(f"Could not remove failed video file {video_output_path}")
    finally:
        # Clean up uploaded file after processing is finished or failed
        if os.path.exists(upload_path):
            try: os.remove(upload_path)
            except OSError: logger.warning(f"Could not remove upload file {upload_path}")
        logger.debug(f"Task {task_id} final status: {tasks.get(task_id)}")


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the upload form."""
    # Clean up old videos when the index page is accessed
    cleanup_old_videos()
    server_busy = is_server_busy()
    return render_template('index.html', server_busy=server_busy)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload, starts background task, returns task ID."""
    try:
        logger.info("Upload request received")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request files: {request.files}")
        logger.info(f"Request form: {request.form}")
        logger.info(f"Request content type: {request.content_type}")
        
        if not request.files:
            logger.error("No files in request")
            return jsonify({"error": "No files were sent in the request."}), 400
            
        if 'file' not in request.files:
            logger.error("No file part in request")
            logger.error(f"Available keys in request.files: {list(request.files.keys())}")
            return jsonify({"error": "No file part in the request. Please ensure you're uploading a file."}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({"error": "No selected file. Please choose a file to upload."}), 400

        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Invalid file type. Only .txt files are allowed."}), 400

        original_filename = secure_filename(file.filename)
        task_id = uuid.uuid4().hex
        upload_filename = f"{task_id}_{original_filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)

        try:
            file.save(upload_path)
            logger.info(f"File uploaded for task {task_id} to {upload_path}")
            logger.info(f"File size: {os.path.getsize(upload_path)} bytes")
        except Exception as e:
            logger.exception(f"Failed to save uploaded file for task {task_id}: {e}")
            return jsonify({"error": f"Error saving uploaded file: {e}"}), 500

        # Store initial task status with creation time
        with tasks_lock:
            tasks[task_id] = {
                "status": "pending",
                "percent": 0,
                "message": "Task queued",
                "created_time": datetime.now(timezone.utc)
            }

        # Define video output filename based on task_id
        video_filename = f"{task_id}_therm_gps_realtime.mp4"

        # Start background thread
        thread = threading.Thread(target=run_video_task, args=(task_id, upload_path, video_filename))
        thread.start()

        # Return task ID to client so it can connect to the stream
        return jsonify({"task_id": task_id}), 202 # 202 Accepted: request accepted, processing not complete
    except Exception as e:
        logger.exception(f"Error in upload_file: {e}")
        return jsonify({"error": f"Error processing the upload: {e}"}), 500


@app.route('/stream/<task_id>')
def stream_status(task_id):
    """Streams task status updates using Server-Sent Events."""
    def generate():
        last_percent = -1
        while True:
            task_info = None
            with tasks_lock:
                task_info = tasks.get(task_id)

            if not task_info:
                logger.warning(f"SSE stream request for unknown task {task_id}")
                yield f"data: {json.dumps({'status': 'error', 'message': 'Task not found.'})}\n\n"
                break

            current_percent = task_info.get('percent', 0)
            current_status = task_info.get('status', 'unknown')

            # Send update only if status changed or percent increased significantly
            # Or always send if completed/error to ensure final state is delivered
            #if current_status != 'processing' or current_percent != last_percent: # Send less frequently
            if True: # Send every time for debugging / more responsive feel
                data_to_send = {
                    "status": current_status,
                    "percent": current_percent,
                    "message": task_info.get('message', ''),
                    "result": task_info.get('result') # Will be None until complete
                }
                yield f"data: {json.dumps(data_to_send)}\n\n" # SSE format: "data: {...}\n\n"
                last_percent = current_percent


            if current_status in ["complete", "error"]:
                logger.info(f"SSE stream for task {task_id} ending with status: {current_status}")
                # Optionally remove task from memory after sending final status?
                # with tasks_lock:
                #     if task_id in tasks: del tasks[task_id]
                break

            time.sleep(0.5) # Poll status every 0.5 seconds

    # Return a streaming response
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# Route to serve static files (videos, etc.) - unchanged
@app.route('/static/<path:filename>')
def static_file(filename):
    """Serves files from the static directory."""
    logger.debug(f"Serving static file request: {filename} from {app.config['STATIC_FOLDER']}")
    try:
        return send_from_directory(app.config['STATIC_FOLDER'], filename)
    except FileNotFoundError:
         logger.error(f"Static file not found: {filename}")
         return "File not found", 404

def is_server_busy():
    """Checks if any video generation tasks are currently processing."""
    with tasks_lock:
        return any(task.get('status') == 'processing' for task in tasks.values())

# --- Main Execution ---
if __name__ == '__main__':
    # Clean up output directory on server start
    cleanup_output_directory()
    # Use threaded=True for development with threading, but for production use Gunicorn/uWSGI
    # app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    # Simpler run for basic testing:
    app.run(debug=False, host='0.0.0.0', port=5000) # Turn debug=False if using threads heavily
