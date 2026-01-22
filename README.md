Internet Firewall Data — IDS Machine Learning Model
📘 Dataset Description

The Internet Firewall Data is a publicly-available dataset from the UCI Machine Learning Repository (Dataset ID 542). It contains real network traffic records captured from a university’s firewall and is widely used for classification tasks in network security and intrusion detection research.
Instances: 65,532
Features: 12
Task: Multiclass Classification
Class Labels:
allow
deny
drop
reset-both
(These represent the action taken by the firewall on a given traffic session.)

📊 Feature Overview

Each row in the dataset represents one firewall log entry, and the following 12 attributes are included:

Feature	Description
Source Port	Port number initiating the connection
Destination Port	Receiving port number
NAT Source Port	Source port after NAT translation
NAT Destination Port	Destination port after NAT translation
Action	Target label (firewall decision)
Bytes	Total bytes transferred
Bytes Sent	Bytes sent by the source
Bytes Received	Bytes received by the destination
Packets	Total number of packets
Elapsed Time (sec)	Duration of the session
pkts_sent	Packets sent by the source
pkts_received	Packets received by the destination
(Attribute list adapted from the dataset documentation)	

There are no missing values in the dataset, and the class label (Action) is used as the target in supervised learning tasks.

🤖 Project: Intrusion Detection System (IDS)

This repository contains a machine learning-based Intrusion Detection System (IDS) trained on the Internet Firewall Data. The main goal is to automatically classify network traffic records as benign or potentially malicious based on the firewall’s historical actions.

🔧 Included ML Components

✔ Data preprocessing and feature scaling
✔ Handling of class imbalance (if applicable)
✔ Model training and evaluation
✔ Performance metrics (Accuracy, F1-score, Precision, Recall, Confusion Matrix)
✔ Trained model checkpoint and prediction interface

🧠 Algorithms Compared

You can include any of the following (based on what you used):

Logistic Regression

Random Forest Classifier

Support Vector Machine

Gradient Boosting

XGBoost / LightGBM

Tip: You can modify this list depending on what models you actually experimented with.

🚀 Usage

Clone the repository

git clone https://github.com/<your-username>/<your-repo>.git


Install dependencies

pip install -r requirements.txt


Train model

python train.py


Run inference

python predict.py --input sample.csv

📈 Results

Provide a summary of your model performance here:

Model	Accuracy	F1-Score	Precision	Recall
Random Forest	98.5%	0.98	0.99	0.97
SVM	96.2%	0.95	0.96	0.94
…	…	…	…	…

(Replace with your actual results.)

📜 Citation

If you use this dataset or code in published work, please cite:

Internet Firewall Data [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5131M
