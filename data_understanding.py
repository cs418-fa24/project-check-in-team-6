# imports
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

"""
    Create a Plotly figure to visualize data with recession/crash periods.
    plotly.graph_objs._figure.Figure: A Plotly figure representing the plot.
"""
def plot_regimes(date_series: pd.Series,
                 data_series: pd.Series, 
                 regime_series: pd.Series,
                 area_name: str,
                 data_name: str,
                 x_axis_title: str,
                 y_axis_title: str,
                 plot_title: str,
                 log_scale: bool = True,
                 line_color: str = 'blue',
                 width: int = 1100,
                 height: int = 550) -> go.Figure:
    
    
    # Initialize figure
    fig = go.Figure()

    # Add recession/crash periods
    fig.add_trace(go.Scatter(
        x=date_series,
        y=regime_series * max(data_series) * 1.01,  # Extend to y_max
        fill='tozeroy',
        fillcolor='rgba(225,225,225,1)',
        line=dict(color='gray', width=0),
        mode='none',
        showlegend=True, 
        name=area_name))

    # Add data series
    fig.add_trace(go.Scatter(
        x=date_series , 
        y=data_series, 
        mode='lines', 
        line_color=line_color, 
        name=data_name))

    # Set x and y axis titles
    fig.update_xaxes(title_text=x_axis_title)
    fig.update_yaxes(title_text=y_axis_title)
    
    if log_scale:
        fig.update_yaxes(type='log')

    # Plot title
    fig.update_layout(title_text=f'<b> {plot_title} </b>', title_x=0.5)

    # Adjust plot size and theme
    fig.update_layout(autosize=False, width=width, height=height, template='simple_white')
    
    return fig

"""
    Downloads and reads the FRED-MD dataset with the specified name.
    Returns:
    pd.DataFrame: DataFrame containing the FRED-MD dataset for the specified date.
"""
def get_fredmd_ds(ds_name: str ='current') -> pd.DataFrame:
    
    # Define url and header for get request, send request
    url = f'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{ds_name}.csv'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise ValueError('Failed to download the dataset!')

    # Read in the CSV file contents
    csv_content = response.content
    fredmd_ds = pd.read_csv(io.BytesIO(csv_content))
    
    # Drop rows with all null values
    fredmd_ds.dropna(how='all', inplace=True)
    
    # rename a column and convert to datetime
    fredmd_ds.rename(columns={'sasdate': 'Date'}, inplace=True)
    fredmd_ds.set_index('Date', inplace=True)

    return fredmd_ds

"""
    Download, extract, and process the FRED-MD dataset's feature description from the FRED-MD Appendix.zip file.
    Returns:
        pd.DataFrame: DataFrame containing FRED-MD feature descriptions, including columns: 'Feature', 'Description', 'Group', 'TCode'.
"""
def get_fredmd_feat_desc(file_name: str = 'FRED-MD_historic_appendix.csv') -> pd.DataFrame:
    
    # Request and download the zip file from FRED-MD Appendix
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get('https://files.stlouisfed.org/files/htdocs/uploads/FRED-MD Appendix.zip', headers=headers)

    if response.status_code == 200:
        with open('FRED-MD_Appendix.zip', 'wb') as file:
            file.write(response.content)

        # Extract the specified CSV file from the zip archive
        with zipfile.ZipFile('FRED-MD_Appendix.zip', 'r') as zip_ref:
            try:
                csv_file_name = next(name for name in zip_ref.namelist() if file_name in name)
            except StopIteration:
                raise ValueError('Failed to find the specified CSV file in the zip archive.')
            zip_ref.extract(csv_file_name, path='extracted_files')

        # Read the extracted CSV file into a DataFrame
        csv_file_path = 'extracted_files/' + csv_file_name
        columns_to_read = ['id', 'tcode', 'fred', 'description', 'group']
        try:
            feat_desc_df = pd.read_csv(csv_file_path, encoding='unicode_escape', usecols=columns_to_read)
        except pd.errors.EmptyDataError:
            raise ValueError('Extracted CSV file is empty or unreadable.')

        # Map group codes to descriptive names
        group_mapping = {
            1: 'Output and Income',
            2: 'Labor Market',
            3: 'Housing',
            4: 'Consumption, Orders and Inventories',
            5: 'Money and Credit',
            6: 'Interest and Exchange Rates',
            7: 'Prices',
            8: 'Stock Market'
        }
        feat_desc_df['Group'] = feat_desc_df['group'].map(group_mapping)

        # Rename columns, set index, and reorder columns
        feat_desc_df.rename(columns={'id': 'ID', 'tcode': 'TCode', 'fred': 'Feature', 'description': 'Description',
                                     'group': 'GroupCode'}, inplace=True)
        feat_desc_df.set_index('ID', inplace=True)
        feat_desc_df = feat_desc_df[['Feature', 'Description', 'Group', 'TCode']]

        # Remove downloaded zip file and extracted files
        os.remove('FRED-MD_Appendix.zip')
        os.remove(csv_file_path)
        shutil.rmtree('extracted_files')

        return feat_desc_df

    else:
        raise ValueError('Failed to download the zip file.')

"""
    Fetches and processes S&P 500 historical daily data to create a monthly OHLC (Open, High, Low, Close) DataFrame.
    Returns:
        pd.DataFrame: A DataFrame with Date index and columns representing monthly OHLC data (Open, High, Low, Close).

    Example:
        >>> sp500_monthly_df = get_sp500_monthly()
"""
def get_sp500_monthly() -> pd.DataFrame:
    
    try:
        # get S&P 500 daily data
        sp500 = yf.Ticker('^GSPC')
        sp500_daily_df = sp500.history(period='max', interval='1d')

        # Convert daily time frame to monthly
        sp500_monthly_df = sp500_daily_df.resample('MS').agg({'Open': 'first', 
                                                              'High': 'max', 
                                                              'Low': 'min', 
                                                              'Close': 'last'})
        # Set "Date" column as the index and convert to datetime type
        sp500_monthly_df.index = pd.to_datetime(sp500_monthly_df.index)
        
        return sp500_monthly_df
    except Exception as e:
        raise RuntimeError("Error fetching or processing S&P 500 data: " + str(e))
        

def fetch_nber_data():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        r = requests.get('http://data.nber.org/data/cycles/business_cycle_dates.json', headers=headers)
        r.raise_for_status()  # Raise an exception for HTTP errors
        return pd.DataFrame(r.json())
    except requests.RequestException as e:
        logging.error("Failed to fetch NBER data: %s", str(e))
        raise

        
def generate_regime_df(df_peak_trough, months_ignored):
    df_peak_trough['peak'] = pd.to_datetime(df_peak_trough['peak'])
    df_peak_trough['trough'] = pd.to_datetime(df_peak_trough['trough'])
    df_peak_trough.dropna(inplace=True)

    start_date = df_peak_trough['peak'].min()
    end_date_max = max(datetime.now() - timedelta(days=months_ignored * 30), df_peak_trough['trough'].max()) 

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

"""
    Generate a DataFrame with monthly NBER business cycle dates.
    
    Returns:
        pd.DataFrame: A DataFrame with Date index and 'Regime' column indicating business cycle regime.
                     The index represents months, and the 'Regime' column indicates the business cycle regime for each
                     month (1 for contraction/recession, 0 for expansion).
    Raises:
        requests.RequestException: If there's an error while fetching data from the NBER website.
        Exception: If there's an error during DataFrame generation.

    Example:
        >>> df = get_nber_cycles(months_ignored=12)
"""
def get_nber_cycles(months_ignored: int = 15) -> pd.DataFrame:


    try:
        df_peak_trough = fetch_nber_data()
        df_regime = generate_regime_df(df_peak_trough, months_ignored)
        return df_regime
    except Exception as e:
        logging.error("Failed to generate NBER business cycle DataFrame: %s", str(e))
        raise
    
"""
    Fetches U.S. recession dates using the 'USREC' code from FRED.
    Returns:
        pd.DataFrame: A DataFrame containing U.S. recession dates with 'Date' as index and 'Regime' as column name.
                    Returns None if data retrieval fails.
"""        
def get_fred_cycles(months_ignored: int = 15) -> pd.DataFrame:
    try:
        # Calculate the start date
        start_date = datetime(1850, 1, 1)
        
        # Calculate the end date by subtracting months_ignored from the current date
        end_date = datetime.now() - pd.DateOffset(months=months_ignored)

        # Fetch U.S. recession dates using the 'USREC' code from FRED
        recession_data = web.DataReader('USREC', 'fred', start_date, end_date)

        # Rename the index and column
        recession_data.index.name = 'Date'
        recession_data.columns = ['EconRegime']

        return recession_data

    except Exception as e:
        print("Failed to fetch NBER data from FRED:", e)
        return None        
