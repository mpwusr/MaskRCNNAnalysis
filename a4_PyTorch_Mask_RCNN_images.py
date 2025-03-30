import os
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import random

# Set device (GPU if available, else CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the pre-trained Mask R-CNN model using the recommended weights parameter
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
model.to(device)

# COCO dataset class names (for Mask R-CNN)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path, threshold):
    """
    Perform instance segmentation on an image and return masks, boxes, and predicted classes.

    Args:
        img_path (str): Path to the input image.
        threshold (float): Confidence threshold for predictions.

    Returns:
        masks, boxes, pred_cls: Segmentation masks, bounding boxes, and predicted class labels.
    """
    # Load the image using PIL and convert to RGB
    img = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format (3 channels)

    # Debug: Print image mode and channels
    print(f"Image mode: {img.mode}, Channels: {len(img.getbands())}")

    # Transform the image to tensor
    transform = T.Compose([T.ToTensor()])  # Converts to [C, H, W], C=3 for RGB
    img_tensor = transform(img).to(device)

    # Debug: Print tensor shape to confirm it's [3, H, W]
    print(f"Image tensor shape: {img_tensor.shape}")

    # Make prediction
    with torch.no_grad():
        prediction = model([img_tensor])

    # Extract predictions
    masks = prediction[0]['masks'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Filter predictions based on the confidence threshold
    pred_cls = [COCO_INSTANCE_CATEGORY_NAMES[labels[i]] for i in range(len(scores)) if scores[i] > threshold]
    masks = masks[scores > threshold]
    boxes = boxes[scores > threshold]

    return masks, boxes, pred_cls


def instance_segmentation_api(img_path, threshold=0.5, save_output=True):
    """
    Process an image for instance segmentation, print the detected objects with segmentation details,
    and visualize the results with segmentation masks.

    Args:
        img_path (str): Path to the input image.
        threshold (float): Confidence threshold for predictions.
        save_output (bool): If True, save the output image instead of displaying it.
    """
    # Get predictions
    masks, boxes, pred_cls = get_prediction(img_path, threshold)

    # Print detailed segmentation information
    print(f"\nSegmentation Details for {os.path.basename(img_path)}:")
    print(f"Number of objects detected: {len(pred_cls)}")
    for i, (cls, box) in enumerate(zip(pred_cls, boxes)):
        print(f"Object {i + 1}:")
        print(f"  Class: {cls}")
        print(f"  Bounding Box: [x1: {box[0]:.2f}, y1: {box[1]:.2f}, x2: {box[2]:.2f}, y2: {box[3]:.2f}]")

    # Visualize the results with segmentation masks
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i, (mask, box, cls) in enumerate(zip(masks, boxes, pred_cls)):
        # Convert mask to binary (0 or 1)
        mask = mask[0]  # Mask shape: [1, H, W] -> [H, W]
        mask = (mask > 0.5).astype(np.uint8)  # Threshold the mask

        # Generate a random color for the mask
        color = [random.randint(0, 255) for _ in range(3)]

        # Overlay the mask on the image
        for c in range(3):
            img[:, :, c] = np.where(mask == 1, img[:, :, c] * 0.5 + color[c] * 0.5, img[:, :, c])

        # Draw bounding box
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        # Add label
        cv2.putText(img, cls, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if save_output:
        # Save the output image
        output_path = os.path.join(os.path.dirname(img_path), f"output_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved output image with segmentation to: {output_path}")
    else:
        # Display the image (closes after 2 seconds)
        cv2.imshow("Instance Segmentation", img)
        cv2.waitKey(2000)  # Wait for 2 seconds
        cv2.destroyAllWindows()


def main():
    # Directory containing images
    # Update this path to your actual image directory
    image_dir = "/Users/michaelwilliams/PycharmProjects/MaskRCNNAnalysis-main/my_images"

    # Ensure the directory exists
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} does not exist. Please update the path.")
        return

    # Get list of image files (case-insensitive extension check)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Debug: Print the number of images found
    print(f"Found {len(image_files)} images in directory: {image_dir}")
    print("Images found:", image_files)

    # Process each image in the directory
    for file_name in image_files:
        img_path = os.path.join(image_dir, file_name)
        print(f"\nProcessing image: {img_path}")

        # Debug: Check the image format before processing
        img = Image.open(img_path)
        print(f"Original image mode: {img.mode}, Channels: {len(img.getbands())}")

        # Process the image
        try:
            # Set save_output=True to save images instead of displaying them
            # Change to save_output=False if you want to display images (closes after 2 seconds)
            instance_segmentation_api(img_path, threshold=0.75, save_output=True)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue  # Continue to the next image even if one fails


if __name__ == "__main__":
    main()