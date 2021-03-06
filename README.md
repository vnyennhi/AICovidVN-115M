# AICovidVN-115M
Solution by Nhi Vo for AICovidVN 115M Challenge: Covid Cough Detection Challenge

This has been tested on HP Laptop Core i5 8GB RAM without GPU using Python 3.8.8
 

## 0. Set up the project folder

Please download all datasets from here, extract and put them in the correct location

Train set: https://bit.ly/aicv115m_public_train

Public Test set: https://bit.ly/aicv115m_public_test

Private Test set: https://bit.ly/aicv115m_private_test

Specifically, all the metadata csv files must be in the "metadata" folder. All the audio files must be in the three audio folders accordingly without any subfolder.

```
train_audio_files_8k
public_test_audio_files_8k
private_test_audio_files_8k
```


Please download the vggish model checkpoint file from the Google Drive link below and put it in the "vggish" folder.
https://drive.google.com/file/d/1D2-mpFV-OSDP_dez5py79Dt80WlmVG4E/view?usp=sharing


You can run all steps below, including the train and predict steps for Private Test Set using the shell script:

```
./run_prediction.sh
```

Open a terminal/bash/shell in the project root folder and follow the below commands

## 1. Create a new virtual environment for the project
```
python -m venv env
```
or
```
python3 -m venv env
```

## 2. Activate the virtual environment

On Mac
```
source env/bin/activate
```

On Windows
```
.\env\Scripts\activate
```

## 3. Install required Python modules
```
pip install -r requirements.txt
```

## 4. Train and predict

Run the "main.py" for all the training and output prediction steps. This should take about 15 minutes to complete.
```
python main.py
```

The final submission is the "results.csv" on the root folder. The trained model and feature arrays are in the folder "model".


If you want to run the prediction step only, use this command
```
python predict.py
```


Note: Since I don't keep a separate "environment.yml", you should update the file paths inside the "main.py" or the "predict.py" file accordingly if you put the metadata and audio files at different locations from those in step 0.



Part of this code is from https://github.com/cam-mobsys/covid19-sounds-kdd20

MIT License (c) 2021
