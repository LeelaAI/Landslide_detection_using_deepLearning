Landslide Detection using Deep Learning

This repository focuses on leveraging Deep Learning (DL) techniques to detect and map landslide events using remote sensing data. By automating the identification of landslide scars, this project aims to support disaster response and environmental monitoring.

000000 Overview
Landslides pose significant risks to infrastructure and human life. Traditional manual mapping is time-consuming; this repository implements state-of-the-art neural networks to:

Identify landslide locations from satellite or aerial imagery.

Segment affected areas to calculate the total land impact.

Streamline the preprocessing of geospatial data for machine learning pipelines.

000000 Tech Stack
Language: Python
Deep Learning Framework: PyTorch / TensorFlow
Geospatial Tools: GDAL, Rasterio, GeoPandas
Libraries: NumPy, Matplotlib, OpenCV
GUI : Tkinter

000000 Dataset
[!NOTE]
It is custom built dataset contain landslide pictures and other dataset contain No landslides pictures.
The models are trained on high-resolution imagery with corresponding binary masks where:
Pixel Value 1: Landslide detected
Pixel Value 0: Non-landslide (background)

000000 Model Architecture
This project implements the following architectures for semantic segmentation:
U-Net: A convolutional network designed for fast and precise segmentation of biomedical images, widely adapted for remote sensing.
ResNet Backbone: Utilizing pre-trained weights for feature extraction to improve accuracy.

000000 Installation
Bash
# Clone the repository
git clone https://github.com/LeelaAI/Landslide_detection_using_deepLearning.git

# Navigate to the directory
cd Landslide_detection_using_deepLearning

# Install dependencies
pip install -r requirements.txt

000000 Future Roadmap
********Integration of multi-temporal imagery (before and after event).
********Incorporating Digital Elevation Models (DEM) to provide topographical context.
********Deployment of a web-based inference tool using Streamlit.
