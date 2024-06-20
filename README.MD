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
Models have been compressed to fit on github. To accelerate inference, either create PhishingDetector object to keep them loaded, or recompress them for yourself.
### Metrics
Main metric measured is f1.
As models have been trained on different datasets, so separately:
1. RandomForestClassifier f1 score = 0.974 (97%)
2. GradientBoostingClassifier f1 score = 0.959 (96%)
## Usage
This project uses poetry with python 3.10. You should have python pre-installed. Make sure to call all commands from the directory with files after cloning.
Alternatively, if you have Poetry installed on your main env, it will automatically create it's own venv, so skip steps 2-4. 
1. Clone repository
~~~
git clone https://github.com/Golgovskiy/Phishing-Detection-ML.git <your folder name>
~~~
2. Enter root folder with shell or open shell in it
~~~
cd <your folder path>
~~~
3. Create new VENV
~~~
python -m venv <your_folder_path>\.venv
~~~
4. Install poetry
~~~
pip install poetry
~~~
5. Then install dependencies
~~~
poetry install 
~~~
6. Finally, to run the script, run
~~~
poetry run python .\console_app.py <your_url>
~~~
To run the UI or API, run 
~~~
poetry run python .\launch_api.py
~~~
## References
The following sources have been reviewed for research:
- https://arxiv.org/html/2401.04820v2
- https://medium.com/intel-software-innovators/detecting-phishing-websites-using-machine-learning-de723bf2f946
- https://paradigmplus.itiud.org/volume3/number3/raj/
- https://github.com/erdemyagcii/NLP-Phishing-Detection
- https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
- https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning/
