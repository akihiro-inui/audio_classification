# Content-based Audio Classification
Plug and play audio classification. It scored 96% for snoring detection as benchmark.  
You can try your own audio dataset.   
This repository support different types of feature extraction methods and classifiers.


## Expert Model  
Currently, k-NN and Logistic Regression are available

## DNN Model  
Currently, MLP, CNN, ResNet(Transfer Learning) are supported  

## How to install  
Make sure to use python 3.6  
`pip install -r backend/requirements.txt  `

## Prepare data  
1. Make folder called `data` with sub-folder indicates categories.  
2. Move audio files into each sub-folders  


## How to run  
Now, you are good to go
Simply, run `python backend/src/main.py`

## Parameter setting, feature extraction methods or different classifiers  
Edit `config/master_config.ini `   
You can select different types of feature extraction and classifier.  
I recommend to set audio length to median of audio files.  