# Phishing detection with machine learning
## This project is using machine learning methods to test wheather the website is phishing or legitimate.
## About
### Definition
A phishing website is a common social engineering method that mimics trustful uniform resource locators (URLs) and webpages. 
### Dataset
This project was trained on two datsets:
1. https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning/data
Some features have been changed to allow real-time inference.
2. https://huggingface.co/datasets/ealvaradob/phishing-dataset
Warning! Webpage code contains numerous viruses and trojans.
### Approach
This project contains two methods:
1. Feature extraction for tabular data classification
2. Bag-of-words NLP classsification.
Both are used and the final answer is combined based on accuracy.
### Models
Multiple models have been tested, and eventually, for the sake of accuracy and speed of inference, two models have been choosen:
1. RandomForestClassifier (sklearn)
2. GradientBoostingClassifier (lightgbm)
Models have been compressed to fit on github.
### Metrics
Main metric measured is f1 (accuracy).
As models have been trained on different datasets it's hard to estimate final accuracy, but separately^
1. RandomForestClassifier accuracy = 0.973847 (97%)
2. GradientBoostingClassifier accuracy = 95876 (96%)
### Issues
The model might be very slightly biased towards calling bare domain urls as phishing, since a lot of features are exctracted from url. (uncertain)
## Usage
This project uses poetry.
Clone repository
Enter root folder with shell
Create new ENV and install poetry
Run
~~~
poetry install
~~~
Then, to run the script, run
~~~
poetry run .\phish-detect\src\main.py <your_url>
## References
The following sources have been reviewed for research:
- https://arxiv.org/html/2401.04820v2
- https://medium.com/intel-software-innovators/detecting-phishing-websites-using-machine-learning-de723bf2f946
- https://paradigmplus.itiud.org/volume3/number3/raj/
- https://github.com/erdemyagcii/NLP-Phishing-Detection
- https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
- https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning/