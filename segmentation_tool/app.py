from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import os
import shutil
import numpy as np
from PIL import Image
from utils.predictor import Predictor
from utils.helpers import (
    blend_mask_with_image,
    save_mask_as_png,
    convert_mask_to_yolo,
)
import torch
from ultralytics import YOLO
import threading
from threading import Lock
import subprocess
import time
import logging
import multiprocessing


# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Define Base Directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Folder structure with absolute paths
UPLOAD_FOLDERS = {
    'input': os.path.join(BASE_DIR, 'static/uploads/input'),
    'segmented_voids': os.path.join(BASE_DIR, 'static/uploads/segmented/voids'),
    'segmented_chips': os.path.join(BASE_DIR, 'static/uploads/segmented/chips'),
    'mask_voids': os.path.join(BASE_DIR, 'static/uploads/mask/voids'),
    'mask_chips': os.path.join(BASE_DIR, 'static/uploads/mask/chips'),
    'automatic_segmented': os.path.join(BASE_DIR, 'static/uploads/segmented/automatic'),
}

HISTORY_FOLDERS = {
    'images': os.path.join(BASE_DIR, 'static/history/images'),
    'masks_chip': os.path.join(BASE_DIR, 'static/history/masks/chip'),
    'masks_void': os.path.join(BASE_DIR, 'static/history/masks/void'),
}

DATASET_FOLDERS = {
    'train_images': os.path.join(BASE_DIR, 'dataset/train/images'),
    'train_labels': os.path.join(BASE_DIR, 'dataset/train/labels'),
    'val_images': os.path.join(BASE_DIR, 'dataset/val/images'),
    'val_labels': os.path.join(BASE_DIR, 'dataset/val/labels'),
    'temp_backup': os.path.join(BASE_DIR, 'temp_backup'),
    'models': os.path.join(BASE_DIR, 'models'),
    'models_old': os.path.join(BASE_DIR, 'models/old'),
}

# Ensure all folders exist
for folder_name, folder_path in {**UPLOAD_FOLDERS, **HISTORY_FOLDERS, **DATASET_FOLDERS}.items():
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Ensured folder exists: {folder_name} -> {folder_path}")

training_process = None


def initialize_training_status():
    """Initialize global training status."""
    global training_status
    training_status = {'running': False, 'cancelled': False}

def persist_training_status():
    """Save training status to a file."""
    with open(os.path.join(BASE_DIR, 'training_status.json'), 'w') as status_file:
        json.dump(training_status, status_file)

def load_training_status():
    """Load training status from a file."""
    global training_status
    status_path = os.path.join(BASE_DIR, 'training_status.json')
    if os.path.exists(status_path):
        with open(status_path, 'r') as status_file:
            training_status = json.load(status_file)
    else:
        training_status = {'running': False, 'cancelled': False}

load_training_status()

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"

# Initialize SAM Predictor
MODEL_CFG = r"C:\codes\sam2\segment-anything-2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"
CHECKPOINT = r"C:\codes\sam2\segment-anything-2\checkpoints\sam2.1_hiera_large.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = Predictor(MODEL_CFG, CHECKPOINT, DEVICE)

# Initialize YOLO-seg
YOLO_CFG = os.path.join(DATASET_FOLDERS['models'], "best.pt")
yolo_model = YOLO(YOLO_CFG)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "app.log"))  # Log to a file
    ]
)


@app.route('/')
def index():
    """Serve the main UI."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image uploads."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file to the input folder
    input_path = os.path.join(UPLOAD_FOLDERS['input'], file.filename)
    file.save(input_path)

    # Set the uploaded image in the predictor
    image = np.array(Image.open(input_path).convert("RGB"))
    predictor.set_image(image)

    # Return a web-accessible URL instead of the file system path
    web_accessible_url = f"/static/uploads/input/{file.filename}"
    print(f"Image uploaded and set for prediction: {input_path}")
    return jsonify({'image_url': web_accessible_url})

@app.route('/segment', methods=['POST'])
def segment():
    """
    Perform segmentation and return the blended image URL.
    """
    try:
        # Extract data from request
        data = request.json
        points = np.array(data.get('points', []))
        labels = np.array(data.get('labels', []))
        current_class = data.get('class', 'voids')  # Default to 'voids' if class not provided

        # Ensure predictor has an image set
        if not predictor.image_set:
            raise ValueError("No image set for prediction.")

        # Perform SAM prediction
        masks, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )

        # Check if masks exist and have non-zero elements
        if masks is None or masks.size == 0:
            raise RuntimeError("No masks were generated by the predictor.")

        # Define output paths based on class
        mask_folder = UPLOAD_FOLDERS.get(f'mask_{current_class}')
        segmented_folder = UPLOAD_FOLDERS.get(f'segmented_{current_class}')

        if not mask_folder or not segmented_folder:
            raise ValueError(f"Invalid class '{current_class}' provided.")

        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(segmented_folder, exist_ok=True)

        # Save the raw mask
        mask_path = os.path.join(mask_folder, 'raw_mask.png')
        save_mask_as_png(masks[0], mask_path)

        # Generate blended image
        blend_color = [34, 139, 34] if current_class == 'voids' else [30, 144, 255]  # Green for voids, blue for chips
        blended_image = blend_mask_with_image(predictor.image, masks[0], blend_color)

        # Save blended image
        blended_filename = f"blended_{current_class}.png"
        blended_path = os.path.join(segmented_folder, blended_filename)
        Image.fromarray(blended_image).save(blended_path)

        # Return URL for frontend access
        segmented_url = f"/static/uploads/segmented/{current_class}/{blended_filename}"
        logging.info(f"Segmentation completed for {current_class}. Points: {points}, Labels: {labels}")
        return jsonify({'segmented_url': segmented_url})

    except ValueError as ve:
        logging.error(f"Value error during segmentation: {ve}")
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        logging.error(f"Unexpected error during segmentation: {e}")
        return jsonify({'error': 'Segmentation failed', 'details': str(e)}), 500

@app.route('/automatic_segment', methods=['POST'])
def automatic_segment():
    """Perform automatic segmentation using YOLO."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    input_path = os.path.join(UPLOAD_FOLDERS['input'], file.filename)
    file.save(input_path)

    try:
        # Perform YOLO segmentation
        results = yolo_model.predict(input_path, save=False, save_txt=False)
        output_folder = UPLOAD_FOLDERS['automatic_segmented']
        os.makedirs(output_folder, exist_ok=True)

        chips_data = []
        chips = []
        voids = []

        # Process results and save segmented images
        for result in results:
            annotated_image = result.plot()
            result_filename = f"{file.filename.rsplit('.', 1)[0]}_pred.jpg"
            result_path = os.path.join(output_folder, result_filename)
            Image.fromarray(annotated_image).save(result_path)

            # Separate chips and voids
            for i, label in enumerate(result.boxes.cls):  # YOLO labels
                label_name = result.names[int(label)]  # Get label name (e.g., 'chip' or 'void')
                box = result.boxes.xyxy[i].cpu().numpy()  # Bounding box (x1, y1, x2, y2)
                area = float((box[2] - box[0]) * (box[3] - box[1]))  # Calculate area

                if label_name == 'chip':
                    chips.append({'box': box, 'area': area, 'voids': []})
                elif label_name == 'void':
                    voids.append({'box': box, 'area': area})

            # Assign voids to chips based on proximity
            for void in voids:
                void_centroid = [
                    (void['box'][0] + void['box'][2]) / 2,  # x centroid
                    (void['box'][1] + void['box'][3]) / 2   # y centroid
                ]
                for chip in chips:
                    # Check if void centroid is within chip bounding box
                    if (chip['box'][0] <= void_centroid[0] <= chip['box'][2] and
                            chip['box'][1] <= void_centroid[1] <= chip['box'][3]):
                        chip['voids'].append(void)
                        break

            # Calculate metrics for each chip
            for idx, chip in enumerate(chips):
                chip_area = chip['area']
                total_void_area = sum([float(void['area']) for void in chip['voids']])
                max_void_area = max([float(void['area']) for void in chip['voids']], default=0)

                void_percentage = (total_void_area / chip_area) * 100 if chip_area > 0 else 0
                max_void_percentage = (max_void_area / chip_area) * 100 if chip_area > 0 else 0

                chips_data.append({
                    "chip_number": int(idx + 1),
                    "chip_area": round(chip_area, 2),
                    "void_percentage": round(void_percentage, 2),
                    "max_void_percentage": round(max_void_percentage, 2)
                })

        # Return the segmented image URL and table data
        segmented_url = f"/static/uploads/segmented/automatic/{result_filename}"
        return jsonify({
            "segmented_url": segmented_url,  # Use the URL for frontend access
            "table_data": {
                "image_name": file.filename,
                "chips": chips_data
            }
        })

    except Exception as e:
        print(f"Error in automatic segmentation: {e}")
        return jsonify({'error': 'Segmentation failed.'}), 500

@app.route('/save_both', methods=['POST'])
def save_both():
    """Save both the image and masks into the history folders."""
    data = request.json
    image_name = data.get('image_name')

    if not image_name:
        return jsonify({'error': 'Image name not provided'}), 400

    try:
        # Ensure image_name is a pure file name
        image_name = os.path.basename(image_name)  # Strip any directory path
        print(f"Sanitized Image Name: {image_name}")

        # Correctly resolve the input image path
        input_image_path = os.path.join(UPLOAD_FOLDERS['input'], image_name)
        if not os.path.exists(input_image_path):
            print(f"Input image does not exist: {input_image_path}")
            return jsonify({'error': f'Input image not found: {input_image_path}'}), 404

        # Copy the image to history/images
        image_history_path = os.path.join(HISTORY_FOLDERS['images'], image_name)
        os.makedirs(os.path.dirname(image_history_path), exist_ok=True)
        shutil.copy(input_image_path, image_history_path)
        print(f"Image saved to history: {image_history_path}")

        # Backup void mask
        void_mask_path = os.path.join(UPLOAD_FOLDERS['mask_voids'], 'raw_mask.png')
        if os.path.exists(void_mask_path):
            void_mask_history_path = os.path.join(HISTORY_FOLDERS['masks_void'], f"{os.path.splitext(image_name)[0]}.png")
            os.makedirs(os.path.dirname(void_mask_history_path), exist_ok=True)
            shutil.copy(void_mask_path, void_mask_history_path)
            print(f"Voids mask saved to history: {void_mask_history_path}")
        else:
            print(f"Voids mask not found: {void_mask_path}")

        # Backup chip mask
        chip_mask_path = os.path.join(UPLOAD_FOLDERS['mask_chips'], 'raw_mask.png')
        if os.path.exists(chip_mask_path):
            chip_mask_history_path = os.path.join(HISTORY_FOLDERS['masks_chip'], f"{os.path.splitext(image_name)[0]}.png")
            os.makedirs(os.path.dirname(chip_mask_history_path), exist_ok=True)
            shutil.copy(chip_mask_path, chip_mask_history_path)
            print(f"Chips mask saved to history: {chip_mask_history_path}")
        else:
            print(f"Chips mask not found: {chip_mask_path}")

        return jsonify({'message': 'Image and masks saved successfully!'}), 200

    except Exception as e:
        print(f"Error saving files: {e}")
        return jsonify({'error': 'Failed to save files.', 'details': str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        saved_images = os.listdir(HISTORY_FOLDERS['images'])
        return jsonify({'status': 'success', 'images': saved_images}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to fetch history: {e}'}), 500


@app.route('/delete_history_item', methods=['POST'])
def delete_history_item():
    data = request.json
    image_name = data.get('image_name')

    if not image_name:
        return jsonify({'error': 'Image name not provided'}), 400

    try:
        image_path = os.path.join(HISTORY_FOLDERS['images'], image_name)
        if os.path.exists(image_path):
            os.remove(image_path)

        void_mask_path = os.path.join(HISTORY_FOLDERS['masks_void'], f"{os.path.splitext(image_name)[0]}.png")
        if os.path.exists(void_mask_path):
            os.remove(void_mask_path)

        chip_mask_path = os.path.join(HISTORY_FOLDERS['masks_chip'], f"{os.path.splitext(image_name)[0]}.png")
        if os.path.exists(chip_mask_path):
            os.remove(chip_mask_path)

        return jsonify({'message': f'{image_name} and associated masks deleted successfully.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to delete files: {e}'}), 500

# Lock for training status updates
status_lock = Lock()

def update_training_status(key, value):
    """Thread-safe update for training status."""
    with status_lock:
        training_status[key] = value

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Handle retrain model workflow."""
    global training_status

    if training_status.get('running', False):
        return jsonify({'error': 'Training is already in progress'}), 400

    try:
        # Update training status
        update_training_status('running', True)
        update_training_status('cancelled', False)
        logging.info("Training status updated. Starting training workflow.")

        # Backup masks and images
        backup_masks_and_images()
        logging.info("Backup completed successfully.")

        # Prepare YOLO labels
        prepare_yolo_labels()
        logging.info("YOLO labels prepared successfully.")

        # Start YOLO training in a separate thread
        threading.Thread(target=run_yolo_training).start()
        return jsonify({'message': 'Training started successfully!'}), 200

    except Exception as e:
        logging.error(f"Error during training preparation: {e}")
        update_training_status('running', False)
        return jsonify({'error': f"Failed to start training: {e}"}), 500
        
def prepare_yolo_labels():
    """Convert all masks into YOLO-compatible labels and copy images to the dataset folder."""
    images_folder = HISTORY_FOLDERS['images']  # Use history images as the source
    train_labels_folder = DATASET_FOLDERS['train_labels']
    train_images_folder = DATASET_FOLDERS['train_images']
    val_labels_folder = DATASET_FOLDERS['val_labels']
    val_images_folder = DATASET_FOLDERS['val_images']

    # Ensure destination directories exist
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)

    try:
        all_images = [img for img in os.listdir(images_folder) if img.endswith(('.jpg', '.png'))]
        random.shuffle(all_images)  # Shuffle the images for randomness

        # Determine split index
        split_idx = int(len(all_images) * 0.8)  # 80% for training, 20% for validation

        # Split images into train and validation sets
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        # Process training images
        for image_name in train_images:
            process_image_and_mask(
                image_name,
                source_images_folder=images_folder,
                dest_images_folder=train_images_folder,
                dest_labels_folder=train_labels_folder
            )

        # Process validation images
        for image_name in val_images:
            process_image_and_mask(
                image_name,
                source_images_folder=images_folder,
                dest_images_folder=val_images_folder,
                dest_labels_folder=val_labels_folder
            )

        logging.info("YOLO labels prepared, and images split into train and validation successfully.")

    except Exception as e:
        logging.error(f"Error in preparing YOLO labels: {e}")
        raise
  
import random

def prepare_yolo_labels():
    """Convert all masks into YOLO-compatible labels and copy images to the dataset folder."""
    images_folder = HISTORY_FOLDERS['images']  # Use history images as the source
    train_labels_folder = DATASET_FOLDERS['train_labels']
    train_images_folder = DATASET_FOLDERS['train_images']
    val_labels_folder = DATASET_FOLDERS['val_labels']
    val_images_folder = DATASET_FOLDERS['val_images']

    # Ensure destination directories exist
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)

    try:
        all_images = [img for img in os.listdir(images_folder) if img.endswith(('.jpg', '.png'))]
        random.shuffle(all_images)  # Shuffle the images for randomness

        # Determine split index
        split_idx = int(len(all_images) * 0.8)  # 80% for training, 20% for validation

        # Split images into train and validation sets
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        # Process training images
        for image_name in train_images:
            process_image_and_mask(
                image_name,
                source_images_folder=images_folder,
                dest_images_folder=train_images_folder,
                dest_labels_folder=train_labels_folder
            )

        # Process validation images
        for image_name in val_images:
            process_image_and_mask(
                image_name,
                source_images_folder=images_folder,
                dest_images_folder=val_images_folder,
                dest_labels_folder=val_labels_folder
            )

        logging.info("YOLO labels prepared, and images split into train and validation successfully.")

    except Exception as e:
        logging.error(f"Error in preparing YOLO labels: {e}")
        raise


def process_image_and_mask(image_name, source_images_folder, dest_images_folder, dest_labels_folder):
    """
    Process a single image and its masks, saving them in the appropriate YOLO format.
    """
    try:
        image_path = os.path.join(source_images_folder, image_name)
        label_file_path = os.path.join(dest_labels_folder, f"{os.path.splitext(image_name)[0]}.txt")

        # Copy image to the destination images folder
        shutil.copy(image_path, os.path.join(dest_images_folder, image_name))

        # Clear the label file if it exists
        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        # Process void mask
        void_mask_path = os.path.join(HISTORY_FOLDERS['masks_void'], f"{os.path.splitext(image_name)[0]}.png")
        if os.path.exists(void_mask_path):
            convert_mask_to_yolo(
                mask_path=void_mask_path,
                image_path=image_path,
                class_id=0,  # Void class
                output_path=label_file_path
            )

        # Process chip mask
        chip_mask_path = os.path.join(HISTORY_FOLDERS['masks_chip'], f"{os.path.splitext(image_name)[0]}.png")
        if os.path.exists(chip_mask_path):
            convert_mask_to_yolo(
                mask_path=chip_mask_path,
                image_path=image_path,
                class_id=1,  # Chip class
                output_path=label_file_path,
                append=True  # Append chip annotations
            )

        logging.info(f"Processed {image_name} into YOLO format.")
    except Exception as e:
        logging.error(f"Error processing {image_name}: {e}")
        raise
  
def backup_masks_and_images():
    """Backup current masks and images from history folders."""
    temp_backup_paths = {
        'voids': os.path.join(DATASET_FOLDERS['temp_backup'], 'masks/voids'),
        'chips': os.path.join(DATASET_FOLDERS['temp_backup'], 'masks/chips'),
        'images': os.path.join(DATASET_FOLDERS['temp_backup'], 'images')
    }

    # Prepare all backup directories
    for path in temp_backup_paths.values():
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    try:
        # Backup images from history
        for file in os.listdir(HISTORY_FOLDERS['images']):
            src_image_path = os.path.join(HISTORY_FOLDERS['images'], file)
            dst_image_path = os.path.join(temp_backup_paths['images'], file)
            shutil.copy(src_image_path, dst_image_path)

        # Backup void masks from history
        for file in os.listdir(HISTORY_FOLDERS['masks_void']):
            src_void_path = os.path.join(HISTORY_FOLDERS['masks_void'], file)
            dst_void_path = os.path.join(temp_backup_paths['voids'], file)
            shutil.copy(src_void_path, dst_void_path)

        # Backup chip masks from history
        for file in os.listdir(HISTORY_FOLDERS['masks_chip']):
            src_chip_path = os.path.join(HISTORY_FOLDERS['masks_chip'], file)
            dst_chip_path = os.path.join(temp_backup_paths['chips'], file)
            shutil.copy(src_chip_path, dst_chip_path)

        logging.info("Masks and images backed up successfully from history.")
    except Exception as e:
        logging.error(f"Error during backup: {e}")
        raise RuntimeError("Backup process failed.")

def run_yolo_training(num_epochs=10):
    """Run YOLO training process."""
    global training_process

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_cfg_path = os.path.join(BASE_DIR, "models/data.yaml")  # Ensure correct YAML path

        logging.info(f"Starting YOLO training on {device} with {num_epochs} epochs.")
        logging.info(f"Using dataset configuration: {data_cfg_path}")

        training_command = [
            "yolo",
            "train",
            f"data={data_cfg_path}",
            f"model={os.path.join(DATASET_FOLDERS['models'], 'best.pt')}",
            f"device={device}",
            f"epochs={num_epochs}",
            "project=runs",
            "name=train"
        ]

        training_process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )

        # Display and log output in real time
        for line in iter(training_process.stdout.readline, ''):
            print(line.strip())
            logging.info(line.strip())
            socketio.emit('training_update', {'message': line.strip()})  # Send updates to the frontend

        training_process.wait()

        if training_process.returncode == 0:
            finalize_training()  # Finalize successfully completed training
        else:
            raise RuntimeError("YOLO training process failed. Check logs for details.")
    except Exception as e:
        logging.error(f"Training error: {e}")
        restore_backup()  # Restore the dataset and masks

        # Emit training error event to the frontend
        socketio.emit('training_status', {'status': 'error', 'message': f"Training failed: {str(e)}"})
    finally:
        update_training_status('running', False)
        training_process = None  # Reset the process


@socketio.on('cancel_training')
def handle_cancel_training():
    """Cancel the YOLO training process."""
    global training_process, training_status

    if not training_status.get('running', False):
        socketio.emit('button_update', {'action': 'retrain'})  # Update button to retrain
        return

    try:
        training_process.terminate()
        training_process.wait()
        training_status['running'] = False
        training_status['cancelled'] = True

        restore_backup()
        cleanup_train_val_directories()

        # Emit button state change
        socketio.emit('button_update', {'action': 'retrain'})
        socketio.emit('training_status', {'status': 'cancelled', 'message': 'Training was canceled by the user.'})
    except Exception as e:
        logging.error(f"Error cancelling training: {e}")
        socketio.emit('training_status', {'status': 'error', 'message': str(e)})

def finalize_training():
    """Finalize training by promoting the new model and cleaning up."""
    try:
        # Locate the most recent training directory
        runs_dir = os.path.join(BASE_DIR, 'runs')
        if not os.path.exists(runs_dir):
            raise FileNotFoundError("Training runs directory does not exist.")

        # Get the latest training run folder
        latest_run = max(
            [os.path.join(runs_dir, d) for d in os.listdir(runs_dir)],
            key=os.path.getmtime
        )
        weights_dir = os.path.join(latest_run, 'weights')
        best_model_path = os.path.join(weights_dir, 'best.pt')

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"'best.pt' not found in {weights_dir}.")

        # Backup the old model
        old_model_folder = DATASET_FOLDERS['models_old']
        os.makedirs(old_model_folder, exist_ok=True)
        existing_best_model = os.path.join(DATASET_FOLDERS['models'], 'best.pt')

        if os.path.exists(existing_best_model):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            shutil.move(existing_best_model, os.path.join(old_model_folder, f"old_{timestamp}.pt"))
            logging.info(f"Old model backed up to {old_model_folder}.")

        # Move the new model to the models directory
        new_model_dest = os.path.join(DATASET_FOLDERS['models'], 'best.pt')
        shutil.move(best_model_path, new_model_dest)
        logging.info(f"New model saved to {new_model_dest}.")

        # Notify frontend that training is completed
        socketio.emit('training_status', {
            'status': 'completed',
            'message': 'Training completed successfully! Model saved as best.pt.'
        })

        # Clean up train/val directories
        cleanup_train_val_directories()
        logging.info("Train and validation directories cleaned up successfully.")

    except Exception as e:
        logging.error(f"Error finalizing training: {e}")
        # Emit error status to the frontend
        socketio.emit('training_status', {'status': 'error', 'message': f"Error finalizing training: {str(e)}"})

def restore_backup():
    """Restore the dataset and masks from the backup."""
    try:
        temp_backup = DATASET_FOLDERS['temp_backup']
        shutil.copytree(os.path.join(temp_backup, 'masks/voids'), UPLOAD_FOLDERS['mask_voids'], dirs_exist_ok=True)
        shutil.copytree(os.path.join(temp_backup, 'masks/chips'), UPLOAD_FOLDERS['mask_chips'], dirs_exist_ok=True)
        shutil.copytree(os.path.join(temp_backup, 'images'), UPLOAD_FOLDERS['input'], dirs_exist_ok=True)
        logging.info("Backup restored successfully.")
    except Exception as e:
        logging.error(f"Error restoring backup: {e}")

@app.route('/cancel_training', methods=['POST'])
def cancel_training():
    global training_process

    if training_process is None:
        logging.error("No active training process to terminate.")
        return jsonify({'error': 'No active training process to cancel.'}), 400

    try:
        training_process.terminate()
        training_process.wait()
        training_process = None  # Reset the process after termination

        # Update training status
        update_training_status('running', False)
        update_training_status('cancelled', True)

        # Check if the model is already saved as best.pt
        best_model_path = os.path.join(DATASET_FOLDERS['models'], 'best.pt')
        if os.path.exists(best_model_path):
            logging.info(f"Model already saved as best.pt at {best_model_path}.")
            socketio.emit('button_update', {'action': 'revert'})  # Notify frontend to revert button state
        else:
            logging.info("Training canceled, but no new model was saved.")

        # Restore backup if needed
        restore_backup()
        cleanup_train_val_directories()

        # Emit status update to frontend
        socketio.emit('training_status', {'status': 'cancelled', 'message': 'Training was canceled by the user.'})
        return jsonify({'message': 'Training canceled and data restored successfully.'}), 200

    except Exception as e:
        logging.error(f"Error cancelling training: {e}")
        return jsonify({'error': f"Failed to cancel training: {e}"}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        for folder in [HISTORY_FOLDERS['images'], HISTORY_FOLDERS['masks_chip'], HISTORY_FOLDERS['masks_void']]:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)  # Recreate the empty folder
        return jsonify({'message': 'History cleared successfully!'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to clear history: {e}'}), 500

@app.route('/training_status', methods=['GET'])
def get_training_status():
    """Return the current training status."""
    if training_status.get('running', False):
        return jsonify({'status': 'running', 'message': 'Training in progress.'}), 200
    elif training_status.get('cancelled', False):
        return jsonify({'status': 'cancelled', 'message': 'Training was cancelled.'}), 200
    return jsonify({'status': 'idle', 'message': 'No training is currently running.'}), 200

def cleanup_train_val_directories():
    """Clear the train and validation directories."""
    try:
        for folder in [DATASET_FOLDERS['train_images'], DATASET_FOLDERS['train_labels'], 
                       DATASET_FOLDERS['val_images'], DATASET_FOLDERS['val_labels']]:
            shutil.rmtree(folder, ignore_errors=True)  # Remove folder contents
            os.makedirs(folder, exist_ok=True)  # Recreate empty folders
        logging.info("Train and validation directories cleaned up successfully.")
    except Exception as e:
        logging.error(f"Error cleaning up train/val directories: {e}")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Required for multiprocessing on Windows
    app.run(debug=True, use_reloader=False)


