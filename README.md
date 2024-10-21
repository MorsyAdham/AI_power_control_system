## Smart AI power control system:
connect multiple appliances to the ports in the control unit, using AI voice recognition and NLP model to understand my commands and control the appliances. For example telling the model which appliance is connected to which port and ask it to control it like turning it on.

## Dataset Creation:
  Using a list containing some home appliances and a dictionary called "command_phrases" containg the intents (classes) with respective commands for the intents.
  using some loops and we create a random dataset with the list and "command_phrases" and assings the respective class to it as well as add some noise to the data to prevent overfitting.
  creates a balanced dataset to make all the intents (classes) equal in size.
  Saves the dataset as a csv file.

## Dataset Visualization:
  Uses various libraries such as "matplotlib", "seaborn", and "wordcloud" to visualize the dataset and it's distripution.

## AI Model:
Has two classes "SimpleDataset" & "PowerControlAIModel".
## 1. SimpleDataset:
  Custom dataset class for handling text and labels.
  Formats the dataset into input_ids, attention_mask, and label.
  Sets max length of the input to 128 to make all data the same.

## 2. PowerControlAIModel:
  Creates a classification model using "BertTokenizer" and "BertForSequenceClassification".
  "create_dataset": create simple dataset using "SimpleDataset" class.
  "train": Train the model using "TrainingArguments" & "Trainer" from "transformers" library to handle all the training process and logs.
  "visualize_training_metrics": Visulaize the model using "matplotlib" for visualizing the model output.
  "load_model": Load the model.
  "predict": Predict and evaluate the output of the model.

## Train Model:
  Uses the classes from the "ai_model" to create and train the model and saves it.

## Predict with Voice:
  Uses "speech_recognition" library and API to understand voice and natural language and turn it into text.
  Uses the "predict" function from "ai_model" to predict the user intent.

