# PlantDetection

An automated pipeline for detecting, segmenting, and measuring plant root systems from digital images (e.g., agar plate cultures). This project leverages deep learning models (**YOLOv8**, **YOLOv11**, and **YOLOv26**) for object detection, followed by traditional computer vision techniques (**OpenCV**) and graph theory (**Breadth-First Search**) to extract morphological traits like root length.

---
## Pipeline
1.  **Dataset:** Trained on a specialized dataset containing **427 digital images** of plants cultivated in Petri dishes.
2.  **Preprocessing & Data Cleaning:**
    * **Image Resizing:** Standardizing image dimensions to optimize GPU memory allocation during training.
    * **Marker Removal:** Automated/manual removal of black permanent marker artifacts (handwritten labels on the plastic dishes) to prevent the models from learning non-biological features.
3.  **Train data split:** Dataset split and evaluated using **Simple split into 70% train 15% validation 15% test ** and **K-Fold cross-validation** strategy to ensure model generalization and avoid overfitting.
4.  **Evaluation of every model:**
5.  **Phenotypic Measurement Methods:**
    * **Bounding Box Estimation:** Quick rough length approximation derived from the spatial boundaries of the predicted bounding boxes.
    * **Segmentation & Skeletonization Pipeline:** Length measurement via instance segmentation mask generation, morphological thinning, skeletonization, and exact tracking using a customized **BFS (Breadth-First Search)** pathfinding algorithm.

---

## 📊 YOLO Architecture Comparison & Insights

During benchmarking, older models (**YOLOv8** and **YOLOv11 (K-Fold)**) outperformed the ultra-modern **YOLOv26** in terms of precision on this dataset.
