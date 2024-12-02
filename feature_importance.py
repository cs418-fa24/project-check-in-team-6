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

"""
    Calculates evaluation metrics for multiple models and visualizes them.
    tuple: Contains the metrics DataFrame and the Plotly figure.
"""
def evaluation_metrics(predictions_df: pd.DataFrame, 
                       model_columns: list,
                       true_label_column: str,
                       display_plot: bool = True,
                       plot_width: int = 700, 
                       plot_height: int = 400) -> tuple:
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


def plot_confusion_matrix(y_true: list, 
                          y_pred: list, 
                          class_names: list = ['Normal', 'Recession'], 
                          title: str = 'RF',
                          width: int = 450,
                          height: int = 450,
                          annotation_text_size: int = 14) -> go.Figure:
    """
    Create an annotated heatmap of the confusion matrix.
    
    Parameters:
        y_true (list): True class labels.
        y_pred (list): Predicted class labels.
        class_names (list, optional): Names of classes. Default is ['Normal', 'Recession'].
        title (str, optional): Plot title. Default is 'RF'.
        width (int, optional): Width of the plot. Default is 450.
        height (int, optional): Height of the plot. Default is 450.
        annotation_text_size (int, optional): Font size of annotations. Default is 14.
    
    Returns:
        go.Figure: A Plotly figure representing the annotated heatmap.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Assign class names
    x = class_names
    y = class_names

    # Convert confusion matrix elements to strings for annotations
    cm_text = [[str(y) for y in x] for x in cm]

    # Create a figure for annotated heatmap
    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=cm_text, colorscale='Blues')
    
    # Adjust x and y axis labels and title
    fig.update_layout(xaxis=dict(title="<b> Predicted Class </b>", side="bottom"), 
                      yaxis=dict(title="<b> Actual Class </b>"),
                      title_text=f'<b> {title} </b>', title_x=0.53)
    
    # Set figure size and theme
    fig.update_layout(autosize=False, width=width, height=height, template='simple_white')
    
    # Change annotation text size
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = annotation_text_size
        
    return fig