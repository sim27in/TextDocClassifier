# Document-Classifier-Flask-App
A Flask web application focused on detecting the category of documents or text. The underlying model was built with a Bidirectional Long Short Term Memory Neural Network with Universal Sentence Encoder for text encoding.

## Introduction
This project aims to provide an interface for classifying documents in PDF or DOC format as well as plain text. The NLP model, trained on a sklearn dataset, demonstrates an accuracy of 93.44% on the test data.

## Methods Used
- Data Cleaning
- Data Preprocessing
- Data Visualisation
- Deep Learning (LSTM-RNN)
- Model Deployent using Flask

## Tools and Technologies Used
- Python
- Numpy, Pandas, NLTK, Matplotlib, Seaborn
- Scikit-Learn, Tensorflow, Keras, Universal Sentence Encoder
- Flask, HTML

## Model Training Description
- Data fetched from sklearn dataset [fetch_20newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
- Data cleaned by removing null values and applying various techniques like lemmatization and removing stopwords
- Text data encoded using [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4), which converted each text row into 512-dimensional vector
- Encoded text converted into numpy array
- Converted imbalanced dataset into balanced dataset using SMOTE technique
- Dataset is divided into an 80:20 ratio as train and test dataset
- Model is trained with Bidirectional LSTM-RNN Neural Network
- Model and encoded labels saved in H5 and joblib format respectively for later use in the Flask app

## Bidirectional LSTM-RNN Characteristics
- Input Shape: (1,512)
- Bidirectional LSTM layer: 512
- Dense Layers: 128 (ReLU), 8 (Softmax)
- Optimizer: 'adam', Loss: 'sparse_categorical_crossentropy'
- Metrics: ['accuracy'], Epochs: 30
- Last Epoch Validation Accuracy: 93.44%

![Accuracy and Loss Graph](https://raw.githubusercontent.com/ShamikRana/Document-Classifier/master/Images/accuracy.png)

## Findings
- The model achieved an accuracy of 93.44% on the test data
- Confusion matrix provides insights into classification performance
![Confusion Matrix](https://raw.githubusercontent.com/ShamikRana/Document-Classifier/master/Images/confusion%20matrix.png)

## Web Application Creation Description
- Three Python files ("predict.py," "read.py," and "visualize.py") were created for creating the TextClassifier class, reading documents, and visualizing the predicted result in the form of a bar graph, respectively.
- These functions were imported into the Flask file "app.py"
- HTML pages in ./templates: "home.html" and "predict.html"
- Uploaded documents are saved in ./static/file for prediction
- Users can either upload "doc" or "pdf" files for prediction or enter document text into the box
- Predicted class and probability are displayed on the predict.html page in the form of a bar graph
### home.html
![Home Page](https://raw.githubusercontent.com/ShamikRana/Document-Classifier/master/Images/home.png)
### predict.html
![Prediction Page](https://raw.githubusercontent.com/ShamikRana/Document-Classifier/master/Images/predict.png)

## How to Use
- Fork this repository to have your own copy
- Clone your copy on your local system
- Install necessary packages into your virtual environment using the command "pip install -r requirements.txt"
- Run "app.py" file
