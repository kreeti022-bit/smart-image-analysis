# 🔍 Smart Image Analysis

A modular **Computer Vision system** built with Python and OpenCV for analyzing images using classical image processing and feature extraction techniques.

## ✨ Features

- 🖼️ Image Preprocessing
- ⚡ Canny, LoG & DoG Edge Detection
- 📐 Hough Line Detection
- 🔹 Harris Corner Detection
- 🔎 SIFT & HOG Feature Extraction
- 📊 Orientation Histogram
- 🧩 Image Segmentation & Contour Detection
- 📦 Object Analysis using Area, Perimeter, Bounding Box & Centroid

## 🛠️ Technologies

**Python · OpenCV · NumPy · Scikit-image · Matplotlib**

## 📂 Project Structure

```text
smart-image-analysis/
├── input/
├── output/
├── src/
├── tests/
├── main.py
├── requirements.txt
├── statement.md
├── README.md
└── LICENSE
``` 
## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
2️⃣ Run the Complete Pipeline
python main.py
3️⃣ Run a Specific Module
python main.py --module edges
python main.py --module features
python main.py --module segmentation
python main.py --module objects
``` 
## 🧪 Testing

Run the test pipeline:

python tests/test_pipeline.py

✅ All implemented modules are tested and validated successfully

## 📊 Sample Results
Analysis	Result
Hough Lines	287
SIFT Keypoints	3838
HOG Features	263736
Significant Regions	19
Orientation Bins	36
## 📁 Output

Processed images and analysis results are automatically stored in the output/ directory.

## 🔮 Future Scope
YOLO-based object detection
Real-time webcam analysis
Semantic segmentation
Object classification
GUI-based visualization
## 👩‍💻 Author

Kreeti Kumari
B.Tech CSE – Artificial Intelligence & Machine Learning
VIT Bhopal | 2026–2027
