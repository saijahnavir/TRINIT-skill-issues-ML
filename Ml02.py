Ml02.py
# %% [markdown]
# # Importing Data
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from tensorflow import keras
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import hamming_loss
import lime
import lime.lime_tabular

# %%


# %%
train_df = pd.read_csv("train.csv")
train_df

# %%
test_df = pd.read_csv("test.csv")
test_df

# %% [markdown]
# # Pre Processing Data

# %%
train_df.shape

# %%
train_df.shape

# %%
train_df.info()

# %%
train_df.isnull().sum()

# %%
train_df.columns

# %%
print(train_df['Commenting'].value_counts())
print(train_df['Ogling/Facial Expressions/Staring'].value_counts())
print(train_df['Touching /Groping'].value_counts())

# %%
train_df.drop_duplicates(subset="Description", keep=False, inplace=True)
train_df.shape

# %%
print(train_df['Commenting'].value_counts())
print(train_df['Ogling/Facial Expressions/Staring'].value_counts())
print(train_df['Touching /Groping'].value_counts())

# %%
print(train_df['Description'][0])

# %%
train_df['Description'] = train_df['Description'].str.lower()
train_df

# %%
import string
print(string.punctuation)
print()

def removePunctuation(text):
  puncFree = "".join([i for i in text if i not in string.punctuation])
  return puncFree

train_df['Description'] = train_df['Description'].apply(removePunctuation)
train_df

# %%
print(train_df['Description'][0])

# %% [markdown]
# # Calculating Hamming score, f1 score and generating predictions.csv and explanations.csv file

# %%
import nltk
nltk.download('punkt')

# %%
import nltk
nltk.download('stopwords')

# %%
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from tensorflow import keras
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import hamming_loss
import shap

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in word_tokenize(text) if word.isalpha() and word not in stop_words])
    return text

# Load data and preprocess text
train_df.drop_duplicates(subset="Description", keep=False, inplace=True)
train_df['Description'] = train_df['Description'].apply(preprocess_text)
test_df['Description'] = test_df['Description'].apply(preprocess_text)

# Tokenization and padding
max_words = 10000
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['Description'])

X_train_sequences = tokenizer.texts_to_sequences(train_df['Description'])
X_test_sequences = tokenizer.texts_to_sequences(test_df['Description'])

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

# Model architecture
embedding_size = 100
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_size, input_length=max_sequence_length),
    GlobalAveragePooling1D(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(3, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train-test split and model training
X = X_train_padded
y = train_df[['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Model evaluation using F1-score for multi-label classification
predictions = model.predict(X_test)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)
f1_score = f1_score(y_test, binary_predictions, average='weighted', zero_division=1)

hamming_score = 1 - hamming_loss(y_test, binary_predictions)
print(f'Hamming Score: {hamming_score}')

print(f'F1 Score: {f1_score}')
print('Classification Report:\n', classification_report(y_test, binary_predictions, zero_division=1))

def preprocess_input_text(text):
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return padded_sequence


predictions_array = []

for text in test_df['Description']:
    input_sequence = preprocess_input_text(text)
    prediction = model.predict(input_sequence, verbose=0)
    threshold = 0.5
    binary_prediction = (prediction > threshold).astype(int)
    predictions_array.append(binary_prediction)
predictions_array = np.array(predictions_array)


print("Predictions Array:")
print(predictions_array)

# Reshape predictions_array to remove the extra dimension
predictions_array_reshaped = predictions_array.reshape(predictions_array.shape[0], predictions_array.shape[2])

# Create a DataFrame with the reshaped predictions_array
predictions_df = pd.DataFrame(predictions_array_reshaped, columns=['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping'])

# Add the original 'Description' column from test_df for reference
predictions_df['Description'] = test_df['Description'].values

# Save predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv' file.")



# %%
# !pip install lime

# %%
def explain_predictions(model, X_test, num_features=10):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_test, mode='classification')
    explanations = []

    for i in range(min(100, len(X_test))):  # Limit to the first 100 entries or less if the test data is smaller
        predict_fn = lambda x: model.predict(x).astype(float)
        exp = explainer.explain_instance(X_test[i], predict_fn, num_features=num_features)
        explanations.append(exp.as_list())

    return explanations

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    X_test_array = X_test[:100]  # Select only the first 100 entries
    explanations = explain_predictions(model, X_test_array)

explanations_df = pd.DataFrame(explanations)
explanations_df.to_csv('explanations.csv', index=False)

print("Explanations for the first 100 entries saved to 'explanations.csv' file.")

# %%
import pickle
with open('ML02.pkl', 'wb') as f:
    pickle.dump(model, f)


