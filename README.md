# imaginary_handwritingBCI

step1.py: time warping + character classification accuracy calculation + t-SNE visualization of classification effect (Note: the core of this step is to install the TWPCA package.https://github.com/ganguli-lab/twpca This study has revised the code and cannot directly upload the revised package library for processing)
step2.py: Use the fitted character template to label the sentence signal through the HMM algorithm
steps 4-5: RNN Training & Inference
step 6: Bigram Language Model
step 7: GPT-2 Rescoring
RNN-LSTM_Decode.py: The core script for the decoding step of the two-level network classifier.
LSTM_rnh.py: LSTM model script (a two-level classification model for identifying r, n, h).
LSTM_XY.py: LSTM model script (a two-level classification model for identifying x, y).
For details, please refer to our paper ‘Research on optimization of brain-controlled typing recognition based on imaginary handwriting’
Disclaimer:
This study is based on the research of Willett et al. For detailed steps, please refer to the paper 'High-performance brain-to-text communication via handwriting'


General
o	python>=3.6
o	tensorflow=1.15
o	numpy (tested with 1.17)
o	scipy (tested with 1.1.0)
o	scikit-learn (tested with 0.20)
