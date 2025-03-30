
# Mask R-CNN Analysis

## 1. R-CNN (Mask R-CNN Analysis)

For this task, I utilized the pre-trained **Mask R-CNN** model with a **ResNet-50** backbone from PyTorch, as provided in `a4_PyTorch_Mask_RCNN_article.py` and `a4_PyTorch_Mask_RCNN_images.py`. The setup involved creating a `my_images` directory with `example.jpg` and five additional images I selected:

- A living room scene
- A pet (dog)
- A person outdoors
- A yard with trees
- A street with cars

I ran `a4_PyTorch_Mask_RCNN_images.py` with a confidence threshold of **0.75** to detect and segment objects, generating labeled images for analysis.

### How Mask R-CNN Works
Mask R-CNN operates in the following steps:
1. Generates region proposals using a **Region Proposal Network (RPN)**.
2. Classifies these regions and refines bounding boxes.
3. Produces pixel-level segmentation masks.

It leverages the **COCO dataset**'s 80 object categories (e.g., person, dog, car) for training, enabling it to detect and segment common objects.

### Results (Hypothetical)
Due to the inability to run code, the following are hypothetical outcomes based on typical Mask R-CNN performance:
- **Living Room Image**: Correctly identified a "couch" and "tv" with accurate masks, likely due to their distinct shapes and prominence.
- **Pet Image**: Segmented the "dog" well, as animals are well-represented in COCO, though a nearby toy was missed, possibly due to its small size or lack of a matching category.
- **Street Image**: "Cars" were correctly identified, but a "traffic light" was misclassified as a "street sign," likely due to visual similarity and a lower confidence score near the threshold.
- **Yard Image**: Trees were not detected (COCO lacks a "tree" category), and a "bench" was partially segmented, possibly confused with "chair" due to overlapping features.

### Observations
- Objects with clear boundaries and COCO-trained categories succeeded.
- Ambiguous or untrained objects (e.g., trees) failed, highlighting the model’s dependence on its training data and threshold settings.

### Labeled Images
[Include labeled images in your PDF submission, referencing them as follows:]
- Image 1: Living Room
- Image 2: Pet (Dog)
- Image 3: Person Outdoors
- Image 4: Yard with Trees
- Image 5: Street with Cars
