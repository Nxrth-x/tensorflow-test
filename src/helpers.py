import os
import tensorflow as tf
from typing import Optional
from pandas import DataFrame, Series


def clear_terminal() -> None:
    os.system("cls || clear")


def make_input_function(
    data_dataframe: DataFrame,
    label_dataframe: Series,
    number_of_epochs: Optional[int] = 10,
    shuffle: Optional[bool] = True,
    batch_size: Optional[int] = 32
):
    def input_function() -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(
            (dict(data_dataframe),
             label_dataframe)
        )

        if shuffle:
            dataset = dataset.shuffle(1000)

        dataset = dataset.batch(batch_size).repeat(number_of_epochs)

        return dataset

    return input_function
