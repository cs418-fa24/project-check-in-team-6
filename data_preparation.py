
"""

1. **DataCleaner Class**:
   - This class provides methods for cleaning a DataFrame by:
     - Removing columns with a number of null values exceeding a specified threshold (`remove_columns_with_nulls`).
     - Removing rows with more than a specified number of null values (`remove_rows_with_nulls`).
     - Forward filling missing values in the DataFrame (`impute_missing_values`).

2. **FeatureEngineer Class**:
   - This class provides methods for transforming and engineering features in a DataFrame:
     - Applying predefined transformations to each feature column based on transformation codes (`apply_feature_transformations`).
     - Adding lag features to the DataFrame for specified lag periods (`add_lag_columns`).
     - Each transformation can include operations like differencing, applying log transformations, and percentage change calculations.

3. **TrendFiltering Class**:
   - This class focuses on analyzing market data to identify market regimes (e.g., recessions or expansions):
     - The `_compute_returns` method calculates daily returns based on a price column (default is 'Close').
     - The `apply_l1_filter` method applies an L1 trend filtering algorithm to the returns data, creating a `MarketRegime` column. This column labels market regimes (0 for expansion and 1 for recession/crash periods) based on the trend filtering results.

The classes allow for easy data cleaning, feature engineering, and analysis of market regimes, which is useful for financial and economic data analysis.
"""


import pandas as pd
import numpy as np
import cvxpy as cp
import warnings

warnings.filterwarnings('ignore')

class DataCleaning:
    """
        remove_columns_with_nulls(): Remove columns with more than max_null_count null values.
        remove_rows_with_nulls(): Remove rows with more than max_null_count null values.
        impute_missing_values(): Impute missing values using forward fill.
    """

    def __init__(self, data):
        """

        df (): The input DataFrame for cleaning.
        """
        self.data = data

    def remove_null_features(self, max_null, inplace=True):
        """
        pd.DataFrame: Cleaned DataFrame with valid columns.
        """
        count_null = self.data.isnull().sum()
        selected_features = count_null[count_null <= max_null].index.tolist()
        data_after_cleaning = self.data[selected_features]

        if inplace:
            self.data = data_after_cleaning

        return data_after_cleaning

    def remove_null_rows(self, max_null, inplace=True):
        """
        pd.DataFrame: Cleaned DataFrame with valid rows.
        """
        data_after_cleaning = self.data[self.data.isnull().sum(axis=1) <= max_null]

        if inplace:
            self.data = data_after_cleaning

        return data_after_cleaning

    def fill_null_obs(self, inplace=True):
        """
        pd.DataFrame: DataFrame with forward-filled missing values.
        """
        # Forward fill null observations
        null_data = self.data.fillna(method='ffill', inplace=False)

        if inplace:
            self.data = null_data

        return null_data
    
    
class FeatureEngineering:
    """
    apply_feature_transformations(): Apply predefined transformations to the features.
    add_lag_columns(): Add lag features for the specified lag periods.
    """


    def __init__(self, data):
        self.data = data
        self.transformation_codes = None

    def __tranfrom_feat__(self, feat_col, trans_code):
        """
        pd.Series: The transformed feature column.
        """
        if trans_code == 1:
            return feat_col
        elif trans_code == 2:
            return feat_col.diff()
        elif trans_code == 3:
            return feat_col.diff(periods=2)
        elif trans_code == 4:
            return feat_col.apply(np.log)
        elif trans_code == 5:
            feat_col = feat_col.apply(np.log).diff(periods=2)
            return feat_col
        elif trans_code == 6:
            feat_col = feat_col.apply(np.log).diff(periods=2)
            return feat_col
        elif trans_code == 7:
            feat_col = feat_col.pct_change().diff()
            return feat_col
