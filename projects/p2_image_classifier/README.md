# Image classification using Deep learning

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I have trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. 

The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

This project uses PyTorch and the torchvision package; the Jupyter Notebook walks through the implementation of the image classifier and shows an example of the classifier's prediction on a test image. The classifier was also converted into a python application which could be run from command line using "train.py" and "predict.py".
