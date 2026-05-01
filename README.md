# Recovery Emotion \& Risk Classifier

#### 

#### Overview



This project is a Natural Language Processing (NLP) system that analyzes user input and classifies it into one of four categories:



* Positive
* Neutral
* Negative
* High Risk (relapse-related language)



It is designed as a recovery-support prototype to detect emotional states and potential relapse signals from text.



##### Tech Stack



* Python
* scikit-learn
* sentence-transformers (all-MiniLM-L6-v2)
* pandas



##### Features



* Real-time text classification
* Confidence scoring
* High-risk keyword detection
* Semantic understanding using embeddings (not just keywords)



##### Example Outputs

Input:



I feel like giving up



Output:



Prediction: High Risk

Confidence: 0.53



Input:



I feel okay today



Output:



Prediction: Neutral

Confidence: 0.41



##### Project Structure



train.py → trains and saves the model

predict.py → loads model and runs predictions

dataset.csv → labeled training data

model.pkl → trained model

label\_encoder.pkl → label mapping



##### How to Run



Install dependencies:

pip install -r requirements.txt



Train the model:

python train.py



Run predictions:=

python predict.py



##### Model Performance



* Train Accuracy: \~0.90+
* Test Accuracy: \~0.60+



##### Future Improvements



* Expand dataset (300–500+ samples)
* Improve neutral vs positive classification
* Deploy as a full web application
* Enhance high-risk detection sensitivity



##### Purpose



This project demonstrates how NLP can be applied to real-world emotional analysis and recovery support systems.

















