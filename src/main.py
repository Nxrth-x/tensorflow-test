import pandas as pd
import tensorflow as tf
from helpers import clear_terminal, make_input_function

# Datasets
dataframe_train = pd.read_csv(
    "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dataframe_evaluation = pd.read_csv(
    "https://storage.googleapis.com/tf-datasets/titanic/eval.csv")

y_train = dataframe_train.pop("survived")
y_evaluation = dataframe_evaluation.pop("survived")

# Columns
CATEGORICAL_COLUMNS = [
    "sex",
    "n_siblings_spouses",
    "parch",
    "class",
    "deck",
    "embark_town",
    "alone"
]

NUMERIC_COLUMNS = [
    "age",
    "fare"
]

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dataframe_train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))


train_input_function = make_input_function(dataframe_train, y_train)

evaluation_input_function = make_input_function(
    dataframe_evaluation,
    y_evaluation,
    number_of_epochs=1,
    shuffle=False
)

linear_estimate = tf.estimator.LinearClassifier(feature_columns)

linear_estimate.train(train_input_function)
accuracy = linear_estimate.evaluate(evaluation_input_function)['accuracy']

result = list(linear_estimate.predict(evaluation_input_function))

clear_terminal()

while True:
    print(f"Average accuracy: {accuracy}")

    number_selected = int(input("\nType in a number to predict: "))
    print("\nPERSON INFO")
    print(dataframe_evaluation.loc[number_selected])
    print(
        f"\nPrediction: {result[number_selected]['probabilities'][1] * 100:.2f}% chance of survival.")
    print(f"Survived? {y_evaluation.loc[number_selected] == 1}")

    if input("\nContinue? [y/N] ").lower() == "n":
        break

    clear_terminal()
