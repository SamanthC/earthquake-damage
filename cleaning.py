# standard imports
from pathlib import Path

# third-party imports
import pandas as pd


class DataSet:
    """One object for both labels and values files

    Args:
    data_dirpath (str) : Data directory pth
        values (str) : Features file
        labels (str) : Labels file

    Attributes:
        values_df (pandas.DataFrame) : Features Dataframe
        labels_df (pandas.Series) : Labels Series
    """

    def __init__(self, values, labels, data_dirpath="Data"):
        self.values_df = pd.read_csv(Path(data_dirpath) / values)
        self.labels_series = pd.read_csv(Path(data_dirpath) / labels)

    def categorical_to_binary(self, columns):
        """Transform categorical columns of a dataframe
        (series) into a binary dataframe

        Args:
            columns (dictionnary) : a dictionnary of column
            labels associated to its prefix in the new dataframe

        """
        for item, prefix in columns.items():
            dummy = pd.get_dummies(self.values_df[item], prefix=prefix)
            self.values_df = pd.concat([self.values_df, dummy], axis=1)
            self.values_df = self.values_df.drop([item], axis=1)
        return self.values_df

    def normalize(self, columns):
        """Normalize columns in order to avoid overweigh of a feature

        Args:
            columns (list) : a list of columns to normalize

        """
        for item in columns:
            self.values_df[item] = (
                self.values_df[item] -
                self.values_df[item].mean()) / self.values_df[item].std()
        return self.values_df

    def drop(self, columns):
        """Drop columns (after correlation analyzis)

        Args:
            columns (list) : a list of columns to drop

        """
        self.values_df = self.values_df.drop(columns, axis=1)
        return self.values_df
