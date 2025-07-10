## 🎥 Real-Time Depth Estimation using MiDaS

This project implements real-time **monocular depth estimation** using Intel's **MiDaS** model. It captures webcam video and overlays a color-mapped depth map next to the original feed for intuitive depth perception.


### 🧠 Built With
* 📦 **PyTorch** — Deep learning framework
* 🎥 **OpenCV** — Webcam capture and visualization
* 🧠 **MiDaS** — Pretrained depth estimation model from Intel
* 💡 **Computer Vision** — Real-time processing pipeline

 
### 📸 Demo
<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*U9BCYDJYITgnCfw0V5OmwA.gif" width="480"/>
</p>


### 📁 Project Structure

├── script_midas.py       # Main Python script for real-time depth estimation

└── README.md             # Project documentation


### ⚙️ How It Works
1. Captures frames from your webcam using OpenCV.
2. Converts each frame to RGB and applies MiDaS preprocessing transforms.
3. Passes the frame through the MiDaS model (MiDaS\_small or DPT).
4. Normalizes and visualizes the depth using `cv2.COLORMAP_INFERNO`.
5. Displays the original and depth-mapped frames side-by-side.


### 🚀 Installation
#### 1. Clone the repository:
git clone https://github.com/yourusername/midas-depth-estimation.git
cd midas-depth-estimation


#### 2. Create and activate a virtual environment (optional):
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


#### 3. Install dependencies:
pip install torch torchvision torchaudio
pip install opencv-python numpy


### ▶️ Run the Script
python script_midas.py

> Press **'q'** to quit the webcam window.


### 💻 Requirements
* Python 3.7+
* OpenCV
* PyTorch (CUDA recommended for better performance)
* Webcam


### 🔒 Disclaimer
* Requires internet connection on first run to download the MiDaS model.
* GPU (CUDA) support is optional but improves real-time performance.


### 👨‍💻 Author
**Arpan Anand Kotian**
📧 [akarpan2005@gmail.com](mailto:akarpan2005@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/arpan-a-k-104897364)

