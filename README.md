This project was done during the participation in the SDU summer school 2025 course - Drones for computer vision applications.

The aim of that two weeks project was to build a drone from scratch.
Fly the drone autonomous over a designated area.
Take images during that flight and classify the objects (animals) seen.

Multiple test flights were conducted to create the training data set.

To gather features a binary images was created. The malahanobis distance and a treshold were used for that.
The binarized imaged was "closed" (erosion of the dilation) to get best features of the objects.

The features are fit to a SVM classifier and then predictions are made from another set of images.
