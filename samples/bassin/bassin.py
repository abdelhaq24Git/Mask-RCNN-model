import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import argparse
from tensorflow.keras.callbacks import TensorBoard

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class BassinConfig(Config):
    NAME = "bassin"
    IMAGES_PER_GPU = 1  # Adjust based on available GPU memory
    NUM_CLASSES = 1 + 1  # Background + basin
    STEPS_PER_EPOCH = 131  # Assuming 143 images and batch size of 1
    DETECTION_MIN_CONFIDENCE = 0.6 # Adjust as needed
class BassinDataset(utils.Dataset):
  def load_basin(self, dataset_dir, subset):
    """Load a subset of the Basin dataset.
    dataset_dir: Root directory of the dataset.
    subset: Subset to load: train or val
    """
    # Add classes. You might have different class names and IDs.
    self.add_class("basin", 1, "irrigation_basin")

    # Train or validation dataset?
    assert subset in ["train", "val"]
    dataset_dir = os.path.join(dataset_dir, subset)

    # Load annotations from your JSON file
    annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))

    # Filter out annotations without bounding box information
    annotations = [a for a in annotations if 'bbox' in a]

    # Add images
    for annotation in annotations:
        # Extract bounding box coordinates
        bbox = annotation['bbox']
        x, y, width, height = bbox

        # Load image size from image file
        image_path = os.path.join(dataset_dir, "train_images", annotation['image_id'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        self.add_image(
            "basin",
            image_id=annotation['image_id'],  # use file name as a unique image id
            path=image_path,
            width=width, height=height,
            bbox=bbox)  # Pass the bounding box information to the dataset
  def load_mask(self, image_id):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    image_info = self.image_info[image_id]
    
    # Check the source of the image
    if image_info["source"] != "bassin":
        return super(self.__class__, self).load_mask(image_id)

    # Load annotations for the image
    annotations = self.image_info[image_id]["annotations"]
    num_instances = len(annotations)

    # Initialize mask array
    mask = np.zeros([image_info["height"], image_info["width"], num_instances],
                    dtype=np.uint8)

    for i, annotation in enumerate(annotations):
        x, y, w, h = annotation["bbox"]  # Extract bbox coordinates
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Set pixel values inside the bounding box to 1
        mask[y:y+h, x:x+w, i] = 1

    # Return masks and class IDs
    return mask.astype(np.bool), np.ones([num_instances], dtype=np.int32)
  def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    if info["source"] == "bassin":
        return info["path"]
    else:
        return super(self.__class__, self).image_reference(image_id)

def train(model, config, train_dataset, val_dataset):
    """Train the model."""
    dataset_train = BassinDataset()
    dataset_train.load_basin(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BassinDataset()
    dataset_val.load_basin(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    tensorboard_callback = TensorBoard(log_dir=args.logs, histogram_freq=1, profile_batch=0)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads',
               callbacks=[tensorboard_callback])

def detect_objects(model, image_path):
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    return r

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Object detection using Mask R-CNN')
    parser.add_argument("command", metavar="<command>", help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False, metavar="/path/to/dataset/", help='Directory of the dataset')
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5", help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False, metavar="path to image", help='Image for object detection')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image, "Argument --image is required for detection"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BassinConfig()
    else:
        class InferenceConfig(BassinConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training" if args.command == "train" else "inference", config=config, model_dir=args.logs)

    # Load weights
    print("Loading weights ", args.weights)
    if args.command == "train":
        model.load_weights(args.weights, by_name=True)
    else:
        model.load_weights(args.weights, by_name=True)

    if args.command == "train":
        dataset_train = BasinDataset()
        dataset_train.load_basin(args.dataset, "train")
        dataset_train.prepare()

        dataset_val = BasinDataset()
        dataset_val.load_basin(args.dataset, "val")
        dataset_val.prepare()

        print("Training network heads")
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')
    elif args.command == "detect":
        detection_results = detect_objects(model, args.image)
        print("Detection results:", detection_results)

if __name__ == '__main__':
    main()
