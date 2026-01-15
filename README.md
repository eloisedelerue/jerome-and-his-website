# A Comparative Grad-CAM Analysis of YOLOv8 on Artistic Representations of Humans
## About This Project
This project explores the intersection of Computer Vision and Art History. The primary objective is to adapt state-of-the-art Object Detection models to identify human figures within classical and modern artworks.
Unlike standard datasets (like COCO) which feature photorealistic images, artworks present unique challenges: stylized anatomy, varying textures (brushstrokes, charcoal), and abstract representations.
The project relies on two main pillars:
- Fine-tuning: Adapting a YOLOv8n model to specialize in detecting humans in art.
- Explainability (XAI): Implementing a modified Grad-CAM algorithm to visualize where the model looks when it makes a detection. Does it focus on faces? Hands? The general silhouette? This project aims to answer these questions.
You can access and test the project without delving into the Jupyter notebooks via the deployed web interface.

## Notebooks
### Key Concepts
#### YOLOv8n
YOLOv8 is a state-of-the-art object detection architecture developed by Ultralytics. It is known for its speed and accuracy, treating object detection as a single regression problem (hence "You Only Look Once").
- We use the "Nano" version, which is the lightest and fastest variant. This makes it efficient for rapid iteration while still being powerful enough to learn complex artistic features.
#### Grad-CAM (Gradient-weighted Class Activation Mapping)
Grad-CAM is a technique used to "open the black box" of Deep Learning models.
- It uses the gradients of the target concept (e.g., the "Human" class) flowing into the final convolutional layer of the network. This produces a coarse localization map highlighting the important regions in the image for predicting the concept.
- While standard Grad-CAM is designed for image classification (is there a cat?), we have adapted it for object detection (where is the cat?). We analyze the gradients relative to specific bounding boxes to understand what features (e.g., a face, a hand) triggered that specific detection.

### Notebooks
#### Jérôme I: Training & Domain Adaptation
The goal of this notebook is to perform Domain Adaptation. It takes a pre-trained YOLOv8n model (initially trained on the COCO dataset of real-world photographs) and fine-tunes it on a specialized dataset of artworks (paintings, sketches, sculptures).
##### Core Tasks
- Data Preparation: Integrating a custom dataset containing various artistic representations of humans.
- Transfer Learning: Leveraging the weights of the "Nano" YOLOv8 model to maintain detection efficiency while specializing the final layers on artistic textures and stylized human forms.
- Optimization: Tuning hyperparameters (learning rate, epochs, and image size) to ensure the model generalizes well across different art periods (e.g., Renaissance vs. Expressionism).
#### Jérôme II: Inference & Comparative Assessment
This notebook acts as the performance benchmark. It allows you to run "side-by-side" detections to visually and statistically compare how the Base YOLOv8n (trained on COCO) and your Fine-Tuned model (from Jérôme I) behave when confronted with the same artworks.
##### Core Tasks
- Dual-Inference Pipeline: Simultaneously runs two different model weights on a single image or a batch of images.
- Qualitative Validation: Visualizing the bounding boxes to see if the Fine-Tuned model successfully detects figures that the Base model missed (e.g., highly stylized, blurry, or fragmented figures).
- False Positive Analysis: Checking if the specialized training reduced "noise" (e.g., the model no longer confuses architectural elements with human silhouettes).
#### Jérôme III: Explainability
This notebook is the "diagnostic center" of the project. It implements a specialized version of Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the decision-making process of the YOLOv8n model. The goal is to identify which specific visual features—such as faces, hands, or silhouettes—the model prioritizes when identifying a human figure in a work of art.
##### Core Tasks
- Grad-CAM Adaptation: Tailored specifically for object detection, this module links the activation maps directly to the predicted bounding boxes rather than the whole image.
- Color Gradient Visualization (Heatmaps): The notebook generates a color-coded heatmap (typically using the JET colormap) overlaid on the artwork.
- Feature Sensitivity Analysis: Allows for the observation of different detection strategies. For example, determining if the model's attention is triggered by the anatomical details of a face or the broader structural geometry of a silhouette.
#### Jérôme IV: Data Augmentation & Dataset Expansion
The effectiveness of any Deep Learning model is directly tied to the quality and quantity of its training data. Jérôme IV is dedicated to Data Augmentation. Its primary goal is to artificially increase the size and diversity of the "Humans in Art" dataset. This notebook is highly recommended (and often essential) if your initial collection of labeled artworks is limited in size.
##### Core Tasks
- Geometric Transformations: Applying rotations, shears, and horizontal flips to help the model recognize human figures regardless of their orientation or composition within the frame.
- Artistic Texture Simulation: Adjusting brightness, contrast, and saturation to mimic different lighting conditions found in galleries or different pigment aging processes.
- Mosaic & Mixup: Implementing advanced YOLOv8 augmentation techniques that combine multiple images into one, forcing the model to detect humans in cluttered or complex artistic backgrounds.
- Format Standardization: Ensuring all augmented images and their corresponding labels (bounding boxes) remain perfectly synchronized and formatted for the YOLOv8 training pipeline.

## Website
