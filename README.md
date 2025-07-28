# Emotion Detection from Uploaded Images
This project aims to develop a Streamlit-based web application that enables users to upload images and accurately detect and classify the emotion in the image using Convolutional Neural Networks (CNNs). The application is designed to be user-friendly and responsive, providing real-time results for emotion detection.

# Table of Contents
1. Project Overview
2. Technologies Used
3. System Architecture
4. Features
5. Usage
6. Model Training
7. Performance Evaluation
8. Ethical Considerations

# Project Overview
This project is built to provide emotion detection from facial expressions in images using Deep Learning. The goal is to integrate machine learning with computer vision to create a system that can detect seven primary emotions: happy, sad, angry, surprised, neutral, disgusted, and fearful. The project uses FER-2013, a dataset containing labeled facial expressions, and implements the solution using Streamlit for a user-friendly interface.

The project consists of:

1. User Interface Development with Streamlit.
2. Emotion Classification through a CNN model trained on the FER-2013 dataset.
3. Performance Optimization for real-time emotion classification.
4. Technologies Used
    Programming Language: Python
    Libraries:
    Streamlit: For building the web application interface.
    PyTorch: For training and evaluating the CNN model.
    Torchvision: For data handling and pre-trained models.
    Dataset: FER-2013 from Kaggle (Facial Expression Recognition dataset).
    System Architecture
The system is divided into several components:

1. User Interface: The Streamlit application allows users to upload an image and get emotion classification results.
2. Emotion Classification: The processed image is passed through a CNN model, trained on the FER-2013 dataset, to predict the emotion.
3. Real-time Performance: The system has been optimized for real-time processing using techniques like model pruning and quantization.
Features
4. Image Upload: Users can upload images in .jpg or .png formats.
5. Emotion Detection: Displays the predicted emotion along with the confidence score.
6. Real-time Feedback: The system provides near-instantaneous emotion classification results.
7. Performance Metrics: Displays the accuracy, precision, recall, and F1 score of the emotion detection model.

# Usage
Open the Streamlit app in your browser.
Upload an image with a clear face.
The system will display the uploaded image and the predicted emotion.

# Model Training
The emotion detection model is trained using the FER-2013 dataset, which consists of over 35,000 labeled facial images. The CNN model has the following architecture:

1. Convolutional Layers: Extract relevant features from the images.
2. Fully Connected Layers: Classify the extracted features into one of the seven emotions.
3. Output Layer: The final output provides the probability distribution across the seven emotions.
4. The training is done using the Adam optimizer and cross-entropy loss for multi-class classification.

# Performance Evaluation
The model was evaluated using the following metrics:

Accuracy: The percentage of correctly classified images.
Precision: The proportion of correct positive predictions among all positive predictions.
Recall: The proportion of correct positive predictions among all actual positive instances.
F1 Score: The harmonic mean of precision and recall.
Project Report
For a detailed explanation of the system design, implementation, performance analysis, and ethical considerations, please refer to the [Project Report](https://docs.google.com/document/d/1LzMS5z27AG2OpO_5vh_Y7bY_Vc58gibGUrjFLlSxyq8/edit?tab=t.0).

# Ethical Considerations
Emotion detection systems must be used responsibly, considering the following:

1. Privacy Concerns: Ensure that user-uploaded images are not stored or shared without user consent. Images should only be processed temporarily and discarded after use.
2. Bias in Emotion Detection: The FER-2013 dataset may not be fully representative of all demographics. The system may exhibit bias in emotion classification for certain age groups, genders, or ethnicities. It is crucial to mitigate such biases by using more diverse datasets and fair training practices.
3. Misuse Potential: Emotion detection can be misused in surveillance or for manipulating users emotionally. Ethical guidelines should be followed to ensure the technology is used responsibly.
