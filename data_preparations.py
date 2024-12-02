
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


    def transform_features(self):
        transform_code = {}
        df_tmp = pd.DataFrame(columns=self.data.columns)
        for column in self.data.columns:
            df_tmp[column] = self.data[column].iloc[1:] 
            transform_code[column] = self.data[column].iloc[0]
        
        self.data = df_tmp
        self.transform_code = transform_code
        
        df_tmp['Date'] = pd.to_datetime(df_tmp['Date'])
        
        transform_data = pd.DataFrame(columns=self.data.columns)
        for column in self.data.columns:
            if column == 'Date':
                transform_data[column] = self.data[column]
            else:
                transform_data[column] = self.__tranfrom_feat__(self.data[column], transform_code[column])
        self.data = transform_data

    def add_lagged_features(self, lag_values):
        for column in self.data.drop(['Date'], axis=1):
            for n in lag_values:
                self.data['{} {}M Lag'.format(column, n)] = self.data[column].shift(n).ffill().values
        self.data.dropna(axis=0, inplace=True)
        return self.data


class TrendFiltering:


    def __init__(self, mkt_data):
        self.mkt_data = mkt_data
    
    def __calc_return__(self, data_col='Close'):
        self.mkt_data['Return'] = self.mkt_data[data_col].pct_change()
        self.mkt_data.dropna(inplace=True)
        self.mkt_data = self.mkt_data[['Close', 'Return']]
        
    def l1_trend_filter(self, lambda_val=0.16):
        """
        pd.DataFrame: DataFrame with market regime labels.
        """
        self.__calc_return__()
        ret_series = self.mkt_data['Return'].values
        n = np.size(ret_series)
        x_ret = ret_series.reshape(n)
        Dfull = np.diag([1]*n) - np.diag([1]*(n-1), 1)
        D = Dfull[0:(n-1),]
        beta = cp.Variable(n)
        lambd = cp.Parameter(nonneg=True)
        
        def tf_obj(x, beta, lambd):
            return cp.norm(x - beta, 2)**2 + lambd * cp.norm(cp.matmul(D, beta), 1)
        
        problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))
        lambd.value = lambda_val
        problem.solve()
        betas_df = pd.DataFrame({'TFBeta': beta.value}, index=self.mkt_data.index)
        betas_df['MktRegime'] = betas_df['TFBeta'].apply(lambda x: 0 if x > 0 else 1)
        self.mkt_data = pd.concat([self.mkt_data, betas_df], axis=1)  
        
        return self.mkt_data[['MktRegime']]