# Data Preparation


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

3. **MarketRegimeAnalyzer Class**:
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

class DataCleaner:
    """
        remove_columns_with_nulls(): Remove columns with more than max_null_count null values.
        remove_rows_with_nulls(): Remove rows with more than max_null_count null values.
        impute_missing_values(): Impute missing values using forward fill.
    """

    def __init__(self, df):
        """

        df (): The input DataFrame for cleaning.
        """
        self.df = df

    def remove_columns_with_nulls(self, max_null_count, inplace=True):
        """
        pd.DataFrame: Cleaned DataFrame with valid columns.
        """
        null_summary = self.df.isnull().sum()
        valid_cols = null_summary[null_summary <= max_null_count].index.tolist()
        cleaned_df = self.df[valid_cols]

        if inplace:
            self.df = cleaned_df

        return cleaned_df

    def remove_rows_with_nulls(self, max_null_count, inplace=True):
        """
        pd.DataFrame: Cleaned DataFrame with valid rows.
        """
        cleaned_df = self.df[self.df.isnull().sum(axis=1) <= max_null_count]

        if inplace:
            self.df = cleaned_df

        return cleaned_df

    def impute_missing_values(self, inplace=True):
        """
        pd.DataFrame: DataFrame with forward-filled missing values.
        """
        filled_df = self.df.fillna(method='ffill', inplace=False)

        if inplace:
            self.df = filled_df

        return filled_df


class FeatureEngineer:
    """
    apply_feature_transformations(): Apply predefined transformations to the features.
    add_lag_columns(): Add lag features for the specified lag periods.
    """

    def __init__(self, df):
        self.df = df
        self.transformation_map = None

    def _apply_single_transformation(self, col, transformation_code):
        """
        pd.Series: The transformed feature column.
        """
        if transformation_code == 1:
            return col
        
        elif transformation_code == 2:
            return col.diff()
        
        elif transformation_code == 3:
            return col.diff(periods=2)
        
        elif transformation_code == 4:
            return col.apply(np.log)
        
        elif transformation_code == 5:
            return col.apply(np.log).diff(periods=2)
        
        elif transformation_code == 6:
            return col.apply(np.log).diff(periods=2)
        
        elif transformation_code == 7:
            return col.pct_change().diff()

    def apply_feature_transformations(self):

        transformation_map = {}

        transformed_df = pd.DataFrame(columns=self.df.columns)
        
        for col in self.df.columns:
            transformed_df[col] = self.df[col].iloc[1:]
            transformation_map[col] = self.df[col].iloc[0]

        self.df = transformed_df
        self.transformation_map = transformation_map

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        result_df = pd.DataFrame(columns=self.df.columns)
        for col in self.df.columns:
            if col == 'Date':
                result_df[col] = self.df[col]
            else:
                result_df[col] = self._apply_single_transformation(self.df[col], transformation_map[col])

        self.df = result_df

    def add_lag_columns(self, lag_values):
        """
        pd.DataFrame: DataFrame with added lagged features.
        """
        for col in self.df.drop(['Date'], axis=1):
            for lag in lag_values:
                self.df[f'{col} {lag}M Lag'] = self.df[col].shift(lag).ffill().values
        self.df.dropna(axis=0, inplace=True)
        return self.df


class MarketRegimeAnalyzer:

    def __init__(self, market_df):

        self.market_df = market_df
    
    def _compute_returns(self, price_column='Close'):
        
        self.market_df['Returns'] = self.market_df[price_column].pct_change()
        self.market_df.dropna(inplace=True)
        self.market_df = self.market_df[['Close', 'Returns']]
        
    def apply_l1_filter(self, lambda_val=0.16):
        """
        pd.DataFrame: DataFrame with market regime labels.
        """
        self._compute_returns()
        returns_values = self.market_df['Returns'].values
        
        n = np.size(returns_values)
        reshaped_returns = returns_values.reshape(n)
        
        D_full = np.diag([1]*n) - np.diag([1]*(n-1), 1)
        D = D_full[0:(n-1), :]
        
        beta_var = cp.Variable(n)
        lambda_param = cp.Parameter(nonneg=True)
        
        def trend_filter_objective(x_vals, beta_var, lambda_param):
            return cp.norm(x_vals - beta_var, 2)**2 + lambda_param * cp.norm(cp.matmul(D, beta_var), 1)
        
        problem = cp.Problem(cp.Minimize(trend_filter_objective(reshaped_returns, beta_var, lambda_param)))
        lambda_param.value = lambda_val
        problem.solve()
        
        trend_filter_betas = pd.DataFrame({'TrendBeta': beta_var.value}, index=self.market_df.index)
        trend_filter_betas['MarketRegime'] = trend_filter_betas['TrendBeta'].apply(lambda x: 0 if x > 0 else 1)
        self.market_df = pd.concat([self.market_df, trend_filter_betas], axis=1)  
        
        return self.market_df[['MarketRegime']]
