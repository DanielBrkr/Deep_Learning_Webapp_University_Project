# Disclaimer: This project, originating from a university group assignment, has been uploaded in its redacted form to exclude any sensitive information such as data, weights, specific files, and configuration settings, rendering the web application non-functional. Its purpose is solely to demonstrate the software engineering skills and project involvement us.


## Table of Contents

1. [General Information](#general-info)
2. [Approach](#approach)
3. [How to run the app using Docker (recommended)](#Docker)
4. [Kubernetes](#kube)
5. [How to run the flask server](#flask)
6. [Model Factory](#model)
7. [Step by Step CNN](#CNN)
8. [Step by Step Classic ML](#classic)

## General Information

This project deals with the classification of car-damages on the basis of images. A label of the four classes: “dent”, “scratch”, “rim” and “other” should be assigned to each image.

This repository includes:

* Jupyter notebook for labeling
* a frontend folder
* a folder for the three different learners (ClassicML Active Learning, Res Net and Xception)
* a folder for pre-processing (includes the feature extraction and selection)
* a folder where the trained models are stored

## Approach

The two main methods employed are: Classic ML models with active learning and CNN models with transfer learning.

## How to run the app using Docker

1. Make sure Docker is installed on your system
2. run `docker login -removed-` and pass your gitlab credentials
3. run `docker run -removed-`
4. Access the image in your browser: [localhost:8888](http://localhost:8888)

In case you do not have access to the privat container registry:

1. build the image locally with `docker build -t -removed- .`
2. run `docker run --name container1 -p 8888:8888 -removed-/test`
3. Access the image in your browser: [localhost:8888](http://localhost:8888)

## Kubernetes

To Deploy the App on the -removed--Kubernetes Cluster:

1. run `kubectl apply -f -removed-
2. deploy the app `kubectl apply -f k8deploy.yml`
3. access the App on [-removed-](http://10.195.8.77:30288/)

## How to run the flask server

Run ‘main.py’
Go to the url shown in the console
If this is not working, try this:
'Add Configuration' in PyCharm on the upper right:
Add new run configuration --> python
Name: for example 'flask.exe'
Script path: Path to 'flask.exe' for example 'C:\Users\user\AppData\Local\Programs\Python\Python39\Scripts\flask.exe'
Parameters: run
Environment variables: PYTHONUNBUFFERED=1;FLASK_APP=main.py;FLASK_DEBUG=1;LANG=EN_EN.UTF-8;FLASK_RUN_PORT=8888
Python Interpreter: your python interpreter path for example 'C:\Users\user\AppData\Local\Programs\Python\Python39\python.exe'
Interpreter Options: nothing
Working Directory: '-removed-' Folder (should be the folder where the ‘main.py’ is) for example -removed-
Add content roots to PYTHONPATH: activate checkbox
Add source roots to PYTHONPATH: activate checkbox
Run this configuration
Go to the url shown in the console

## Model Factory

In the *model_factory* models are constructed based on the users preferences.

* _class ResNet50V2:_ A simple but efficient CNN model with only one hidden dense layer.
* _class XceptionTransfer:_ This uses the best perfoming Xception model architecture
* _class ClassicalML:_ a supervised classification model

## CNN

1. Preprocessing (as input layers before the base mode (gray scaling, input scaling and other normalization procedures)
2. Data splitting (training set, validation set, test set and a refinement set)
3. Classifier head training (training the model on batch sizes of 32 and a duration of 20 epochs)
4. Model fine-tuning (take best performing models from 4. for fine-tuning)
5. Final model testing (testing regarding accuracy, cohens kappa precision, recall and f1 score)

## Classic ML

1. Preprocessing (Train/Test/Validation Split, Feature Extraction, Feature Selection, Dimensionality Reduction)
2. Select Model (K- Nearest Neighbour, Logistic Regression, Support Vector Machine, Random Forest, Ensemble Model)
3. Training via Active Learning
4. Final Model testing
