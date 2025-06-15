Animal Image Classifier
using MobileNetV2 transfer-learning

A project for classifying animals species in images using convolutional neural networks
and transfer learning with MobileNetV2. In this folder also is our journey, arriving at our
best model Model 7, included, starting from our Model 0.

Project overview

This project applies transfer learning to eﬃciently classify images of 10 animal types.
We used the MobileNetV2 architecture, pre-trained on ImageNet, and fine-tuned it for our
custom data set to achieve high accuracy with limited training time and resources.

Data set

Source: Animal-10 data set
Size: approx. 28,000 images
Classes: dog, cat, horse, cow, elephant, chicken, sheep, butterflies, spider, squirrel

Model architecture

Base model: MobileNetV2 (pre trained on ImageNet)
Custom top layers: GlobalAveragePooling2D
Dropout(0.3,0.5)
Dense(64, activation='relu')
Dense(10, activation='softmax')

Training Details

Loss function: sparse_categorical_crossentropy
Optimizer: Adam (tested 1e-5 to 3e-3)
Callbacks:
• EarlyStopping (patience=10)
• ReduceLROnPlateau
• ModelCheckpoint
Data split: 70% training, 15% validation, 15% test

Results

Best accuracy: 96% on validation data
Eﬀective use of: dropout to reduce overfitting
learning rate scheduling
fine-tuning for better generalisation
Key advantages: lightweight and fast with MobileNetV2, works well with limited data,
achieves high accuracy with transfer-learning, generalisable to other image classification
tasks

Team: Jean-Denis, Tejal, Adrianna, Darius
