# Data Understanding



"""
This code defines several functions for fetching, cleaning, and visualizing economic data, specifically for market regimes and business cycles.

1. **plot_market_regimes()**:
   - Uses Plotly to create an interactive plot showing economic regimes (e.g., recessions) and market data over time.
   - It visualizes market data and highlights recession/crash periods on the plot using shaded areas.

2. **fetch_fredmd_dataset()**:
   - Fetches the FRED-MD dataset (a collection of macroeconomic time series) from the Federal Reserve Bank of St. Louis.
   - It loads the dataset into a Pandas DataFrame, drops rows with all NaN values, and formats the 'Date' column.

3. **fetch_fredmd_feature_descriptions()**:
   - Downloads and extracts a zip file containing the feature descriptions for the FRED-MD dataset.
   - Parses the extracted CSV file into a DataFrame, providing metadata such as feature names, descriptions, and groupings.

4. **fetch_sp500_monthly_data()**:
   - Retrieves the monthly historical data for the S&P 500 index using Yahoo Finance.
   - Resamples the daily data to monthly data (OHLC format: Open, High, Low, Close).

5. **retrieve_nber_cycle_data()**:
   - Fetches the NBER business cycle data (peak and trough dates of business cycles) in JSON format and converts it into a Pandas DataFrame.

6. **create_market_regime_df()**:
   - Generates a DataFrame with monthly market regimes (expansion or recession periods) based on the NBER peak and trough data.
   - The regimes are labeled as '0' for expansion and '1' for recession.

7. **fetch_nber_business_cycles()**:
   - A wrapper function that calls `retrieve_nber_cycle_data()` and `create_market_regime_df()` to fetch and create a monthly business cycle DataFrame.
   - Returns a DataFrame that indicates the market regime (0 or 1) for each month.

8. **fetch_fred_recession_cycles()**:
   - Retrieves recession data from FRED using the 'USREC' series, which indicates whether the U.S. economy was in a recession during a given period.
   - The data is cleaned and formatted as a DataFrame with a date index and 'EconRegime' column.

The functions allow for fetching, cleaning, and visualizing economic and market data for analyzing business cycles and economic regimes.
"""



import pandas as pd
import numpy as np
import pandas_datareader.data as web
import requests
import io
from datetime import datetime, timedelta
import yfinance as yf
import logging
import zipfile
import shutil
import os
import plotly.graph_objects as go


def plot_market_regimes(date_values: pd.Series,
                        data_values: pd.Series, 
                        regime_values: pd.Series,
                        recession_area_label: str,
                        data_label: str,
                        x_axis_label: str,
                        y_axis_label: str,
                        plot_title_text: str,
                        log_scale_flag: bool = True,
                        line_color_code: str = 'blue',
                        fig_width: int = 1100,
                        fig_height: int = 550) -> go.Figure:
    """
    plotly.graph_objs._figure.Figure: The Plotly figure object.
    """
    
    # Initialize the plot
    fig = go.Figure()

    # Add recession/crash periods
    fig.add_trace(go.Scatter(
        x=date_values,
        y=regime_values * max(data_values) * 1.01,  # Adjust y-axis range
        fill='tozeroy',
        fillcolor='rgba(225,225,225,1)',
        line=dict(color='gray', width=0),
        mode='none',
        showlegend=True, 
        name=recession_area_label))

    # Add the market data series
    fig.add_trace(go.Scatter(
        x=date_values, 
        y=data_values, 
        mode='lines', 
        line_color=line_color_code, 
        name=data_label))

    # Set the titles for axes
    fig.update_xaxes(title_text=x_axis_label)
    fig.update_yaxes(title_text=y_axis_label)
    
    if log_scale_flag:
        fig.update_yaxes(type='log')

    # Add plot title
    fig.update_layout(title_text=f'<b>{plot_title_text}</b>', title_x=0.5)

    # Customize plot appearance
    fig.update_layout(autosize=False, width=fig_width, height=fig_height, template='simple_white')
    
    return fig


def fetch_fredmd_dataset(ds_identifier: str = 'current') -> pd.DataFrame:
    """
    pd.DataFrame: The FRED-MD dataset as a DataFrame.
    """
    
    # Construct the URL for the dataset and send a request
    url = f'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{ds_identifier}.csv'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise ValueError('Failed to download dataset!')

    # Load the dataset into a pandas DataFrame
    csv_content = response.content
    fredmd_data = pd.read_csv(io.BytesIO(csv_content))
    
    # Drop rows where all values are NaN
    fredmd_data.dropna(how='all', inplace=True)
    
    # Rename the column and convert 'Date' to datetime format
    fredmd_data.rename(columns={'sasdate': 'Date'}, inplace=True)
    fredmd_data.set_index('Date', inplace=True)

    return fredmd_data


def fetch_fredmd_feature_descriptions(file_name: str = 'FRED-MD_historic_appendix.csv') -> pd.DataFrame:
    """
    pd.DataFrame: A DataFrame containing feature descriptions from the FRED-MD dataset.
    """
    
    # Fetch and unzip the appendix file from FRED
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get('https://files.stlouisfed.org/files/htdocs/uploads/FRED-MD Appendix.zip', headers=headers)

    if response.status_code == 200:
        with open('FRED-MD_Appendix.zip', 'wb') as file:
            file.write(response.content)

        # Extract the specified CSV file
        with zipfile.ZipFile('FRED-MD_Appendix.zip', 'r') as zip_ref:
            try:
                csv_file_name = next(name for name in zip_ref.namelist() if file_name in name)
            except StopIteration:
                raise ValueError('File not found in the zip archive.')
            zip_ref.extract(csv_file_name, path='extracted_files')

        # Read the extracted file into a DataFrame
        csv_file_path = f'extracted_files/{csv_file_name}'
        columns_to_read = ['id', 'tcode', 'fred', 'description', 'group']
        try:
            feat_desc_df = pd.read_csv(csv_file_path, encoding='unicode_escape', usecols=columns_to_read)
        except pd.errors.EmptyDataError:
            raise ValueError('The extracted CSV file is empty or unreadable.')

        # Map group codes to their descriptive names
        group_mapping = {
            1: 'Output and Income',
            2: 'Labor Market',
            3: 'Housing',
            4: 'Consumption, Orders, and Inventories',
            5: 'Money and Credit',
            6: 'Interest and Exchange Rates',
            7: 'Prices',
            8: 'Stock Market'
        }
        feat_desc_df['Group'] = feat_desc_df['group'].map(group_mapping)

        # Clean up and return the DataFrame
        feat_desc_df.rename(columns={'id': 'ID', 'tcode': 'TCode', 'fred': 'Feature', 'description': 'Description',
                                     'group': 'GroupCode'}, inplace=True)
        feat_desc_df.set_index('ID', inplace=True)
        feat_desc_df = feat_desc_df[['Feature', 'Description', 'Group', 'TCode']]

        # Remove the temporary files
        os.remove('FRED-MD_Appendix.zip')
        os.remove(csv_file_path)
        shutil.rmtree('extracted_files')

        return feat_desc_df

    else:
        raise ValueError('Failed to download the zip file.')


def fetch_sp500_monthly_data() -> pd.DataFrame:
    """
    pd.DataFrame: A DataFrame containing the S&P 500 index with monthly OHLC data.
    """
    
    try:
        # Fetch daily data for the S&P 500
        sp500 = yf.Ticker('^GSPC')
        sp500_daily_df = sp500.history(period='max', interval='1d')

        # Resample the data into monthly OHLC
        sp500_monthly_df = sp500_daily_df.resample('MS').agg({'Open': 'first', 
                                                              'High': 'max', 
                                                              'Low': 'min', 
                                                              'Close': 'last'})
        # Set the 'Date' column as the index and convert to datetime
        sp500_monthly_df.index = pd.to_datetime(sp500_monthly_df.index)
        
        return sp500_monthly_df
    except Exception as e:
        raise RuntimeError("Error fetching or processing S&P 500 data: " + str(e))
        

def retrieve_nber_cycle_data():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        r = requests.get('http://data.nber.org/data/cycles/business_cycle_dates.json', headers=headers)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except requests.RequestException as e:
        logging.error("Failed to fetch NBER data: %s", str(e))
        raise

        
def create_market_regime_df(df_peak_trough, ignored_months):
    df_peak_trough['peak'] = pd.to_datetime(df_peak_trough['peak'])
    df_peak_trough['trough'] = pd.to_datetime(df_peak_trough['trough'])
    df_peak_trough.dropna(inplace=True)

    start_date = df_peak_trough['peak'].min()
    end_date_max = max(datetime.now() - timedelta(days=ignored_months * 30), df_peak_trough['trough'].max()) 

    date_range = pd.date_range(start=start_date.replace(day=1), end=end_date_max.replace(day=1), freq='MS') 
    df_regime = pd.DataFrame(index=date_range)                                                           
    df_regime.index.name = 'Date'
    df_regime['Regime'] = 0

    for _, row in df_peak_trough.iterrows():
        peak = row['peak']
        trough = row['trough']
        mask = (df_regime.index >= peak) & (df_regime.index <= trough)
        df_regime.loc[mask, 'Regime'] = 1

    for trough in df_peak_trough['trough']:
        df_regime.at[trough.replace(day=1), 'Regime'] = 0

    return df_regime


def fetch_nber_business_cycles(ignored_months: int = 15) -> pd.DataFrame:
    """
    pd.DataFrame: A DataFrame with monthly business cycle dates.
    """

    try:
        df_peak_trough = retrieve_nber_cycle_data()
        df_regime = create_market_regime_df(df_peak_trough, ignored_months)
        return df_regime
    except Exception as e:
        logging.error("Failed to generate NBER business cycle DataFrame: %s", str(e))
        raise
        
def fetch_fred_recession_cycles(ignored_months: int = 15) -> pd.DataFrame:
    """
    pd.DataFrame: A DataFrame with recession dates from FRED.
    """
    try:
        # Calculate the start and end dates
        start_date = datetime(1850, 1, 1)
        end_date = datetime.now() - pd.DateOffset(months=ignored_months)

        # Fetch the recession data from FRED
        recession_data = web.DataReader('USREC', 'fred', start_date, end_date)

        # Clean up the DataFrame
        recession_data.index.name = 'Date'
        recession_data.columns = ['EconRegime']

        return recession_data

    except Exception as e:
        print("Failed to fetch NBER data from FRED:", e)
        return None
