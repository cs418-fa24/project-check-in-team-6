import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score


def plot_feature_importance(features_df: pd.DataFrame,
                            groups_df: pd.DataFrame,
                            chart_title: str,
                            bar_chart_title: str,
                            pie_chart_title: str,
                            start_index: int, 
                            max_features: int) -> go.Figure:

    return fig


def evaluation_metrics(predictions_df: pd.DataFrame, 
                       model_columns: list,
                       true_label_column: str,
                       display_plot: bool = True,
                       plot_width: int = 700, 
                       plot_height: int = 400) -> tuple:
    """
    Calculates evaluation metrics for multiple models and visualizes them.
        tuple: Contains the metrics DataFrame and the Plotly figure.
    """
    # List of evaluation functions
    eval_metrics = [accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score]
    results = {}

    # Calculate metrics for each model
    for model in model_columns:
        # Convert predicted probabilities to binary outcomes (0 or 1)
        predicted_labels = np.where(predictions_df[model] > 0.5, 1, 0)
        true_labels = predictions_df[true_label_column]
        # Calculate metrics for the model
        results[model] = [metric(true_labels, predicted_labels) for metric in eval_metrics]

    # Store results in a DataFrame
    metrics_df = pd.DataFrame(results, index=['Accuracy', 'B-Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC'])

        
    return metrics_df


