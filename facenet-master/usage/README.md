# FaceNet Usage Guidelines

This "usage" folder demonstrates step-by-step guidelines on how to setup real-time facial recognition with a webcam using the FaceNet framework by David Sandberg. https://github.com/davidsandberg/facenet

### Pre-requisites:

	1. Install PIP package dependencies
	   Go to facenet directory and run "pip install -r requirements.txt"
	2. Install facenet 
	   pip install facenet
	3. Download the models, unzip and put inside facenet\usage\models
	   https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
	   https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz

### Training:

	1. Add photos in facenet\usage\datasets\train
	   train\person1\person1_01.jpg
	   train\person1\person1_02.jpg
	   ...
	   train\person1\person1_XX.jpg
	   train\person2\person2_01.jpg
	   train\person2\person2_02.jpg
	   ...
	   train\person2\person2_YY.jpg
	2. Execute train.bat
	   ::python classifier.py TRAIN datasets\train models\20180408-102900\20180408-102900.pb models\datasets_classifier.pkl --batch_size 1000
	   python classifier.py TRAIN datasets\train models\20180402-114759\20180402-114759.pb models\datasets_classifier.pkl --batch_size 1000

### Testing:

	1. Add photos in facenet\usage\datasets\test
	   test\person1\person1_01.jpg
	   test\person1\person1_02.jpg
	   ...
	   test\person1\person1_WW.jpg
	   test\person2\person2_01.jpg
	   test\person2\person2_02.jpg
	   ...
	   test\person2\person2_ZZ.jpg
	2. Execute test.bat
	   ::python classifier.py CLASSIFY datasets\test models\20180408-102900\20180408-102900.pb models\datasets_classifier.pkl
	   python classifier.py CLASSIFY datasets\test models\20180402-114759\20180402-114759.pb models\datasets_classifier.pkl

### Testing w/a webcam:
	1. Connect a webcam
	2. Execute test_webcam.bat
	   ::python real_time_face_recognition.py --model models/20180408-102900 --classifier models/datasets_classifier.pkl
	   python real_time_face_recognition.py --model models/20180402-114759 --classifier models/datasets_classifier.pkl
