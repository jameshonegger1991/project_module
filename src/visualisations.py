import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import textwrap
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.config import (
    PLOTS_DIR,
)

def export_histograms(df, title = "Histogram"):
    """
    Save histograms for all numeric columns to 'plots/univariate_plots/histograms'.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate histograms.")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to generate histograms.")
        return
    
    os.makedirs("plots/univariate_plots/histograms", exist_ok=True)
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        plt.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.title(f"{title}: {col}", fontsize=16)           
        plt.xlabel('Value', fontsize=14)                   
        plt.ylabel('Frequency', fontsize=14)               
        plt.xticks(fontsize=12)                            
        plt.yticks(fontsize=12)                            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        #plt.show()
        plt.savefig(f"plots/univariate_plots/histograms/{col}_{title}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"{len(numeric_cols)} histograms saved to plots/univariate_plots/")

def export_violin_plots(df, title = "Violin Plot"):
    """
    Save violin plots for all numeric columns to 'outputs/plots/univariate_plots/violin_plots/'.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate violin plots.")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to generate violin plots.")
        return
    
    os.makedirs("outputs/plots/univariate_plots/violin_plots", exist_ok=True)
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        
        # One violin plot per feature
        sns.violinplot(data=df[col].dropna(), orient='h')
        
        plt.title(f"{title}: {col}", fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"outputs/plots/univariate_plots/violin_plots/{col}_{title}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"{len(numeric_cols)} violin plots saved to outputs/plots/univariate_plots/violin_plots/")

def export_barplots(df, title = "Bar Plot"):
    """
    Generate bar plots for all categorical features and save them to 'outputs/plots/univariate_plots/barplots/'.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate bar plots.")
        return

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        print("No categorical columns found to generate bar plots.")
        return

    
    folder_path = "outputs/plots/univariate_plots/barplots"
    os.makedirs(folder_path, exist_ok=True)
    
    for col in categorical_cols:
        
        if df[col].isnull().all():
            print(f"Column '{col}' contains only missing values. Skipping bar plot.")
            continue

        counts = df[col].value_counts()
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts.index, y=counts.values)
        
        plt.title(f"{title}: {col}", fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(f"{folder_path}/{col}_{title}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"{len(categorical_cols)} bar plots saved to {folder_path}/")

def display_histograms(df, title = "Histograms for all numeric features"):
    """
    Display a grid of histograms for all numeric columns.
    """ 
    if df.empty:
        print("The DataFrame is empty. Cannot generate histograms.")
        return

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to display histograms.")
        return

    # `hist` returns a 2D array of axes (subplots), even if there's only one row or column.
    axes = df[numeric_cols].hist(figsize=(22, 18), bins=30, xlabelsize=8, ylabelsize=8)

    # `.ravel()` is used to flatten this 2D array into a 1D array, making it easier to iterate through each subplot.
    for ax in axes.ravel():
        ax.set_title(ax.get_title(), fontsize=8)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.5)
    plt.subplots_adjust(hspace=0.9, wspace=0.4)
    plt.show()

def display_violin_plots(df, title = "Violin Plots - All Numeric Features"):
    """
    Displays violin plots for all numeric columns in the DataFrame. 
    Inspiration: https://www.datasciencebyexample.com/2022/05/15/2022-05-15-1/ 
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate violin plots.")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to display violin plots.")
        return

    n_cols = 4 

    # The formula (len(list) + n_cols - 1) // n_cols is a common way to calculate
    # the number of rows needed to fit all items in a grid, ensuring all items are covered.
    # Source: https://discuss.python.org/t/integer-ceiling-divide/91269 
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols 
    

    # Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4.5)) # Adjusted figsize for better layout
    
    # Flatten the axes array to easily iterate over all subplots, regardless of grid dimensions.
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    axes = axes.flatten()
    
    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j # Calculate the current index in the flattened axes array
            if index < len(numeric_cols):
                col = numeric_cols[index]
                # Create a horizontal violin plot for the current column.
                # Source: https://seaborn.pydata.org/generated/seaborn.violinplot.html
                sns.violinplot(data=df[col], orient='h', ax=axes[index]) 
                axes[index].set_title(col, fontsize=8) 
                axes[index].set_xlabel('') 
                axes[index].tick_params(labelsize=8) 
            else:
                # If there are more subplots than numeric columns, turn off the unused subplots.
                axes[index].axis('off')
    
    plt.suptitle(title, fontsize=16) 
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.5) # Adjust layout to prevent titles/labels from overlapping. The rect parameter leaves space for suptitle.
    plt.subplots_adjust(hspace=1.0, wspace=0.4) # Adjust the height and width spacing between subplots.
    plt.show()

def display_barplots(df, title = "Bar Plots - All Categorical Features"):
    """
    Displays bar plots for all categorical columns in the DataFrame.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate bar plots.")
        return

    #This function is based on the same model as "display_violin_plots"
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not cat_cols:
        print("No categorical columns found to display bar plots.")
        return

    n_cols = 2
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    # np.asarray is used here to ensure 'axes' is a NumPy array, even if subplots returns a single Axes object
    axes = np.asarray(axes).flatten() 
    
    for i in range(len(cat_cols)):
        col = cat_cols[i]
        counts = df[col].value_counts()
        
        # Replace spaces with newlines in labels for better readability on the x-axis
        labels = [label.replace(' ', '\n') for label in counts.index]
        
        axes[i].bar(labels, counts.values)
        axes[i].set_title(col)
        axes[i].set_ylabel('Count')
        axes[i].tick_params(axis='x', labelsize=8, rotation=90)
        axes[i].set_xticklabels(labels, ha='right')
    
    for i in range(len(cat_cols), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def display_spearman_correlation_matrix(df, title = 'Spearman Correlation Matrix'):
    """
    Displays the Spearman correlation matrix for all numeric variables.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate Spearman correlation matrix.")
        return

    # Calculate the Spearman correlation matrix
    spearman_matrix = df.corr(method='spearman')
    # Sort variables by correlation with the target
    corr_asc = spearman_matrix["PV1MATH"].abs().sort_values(ascending=False).index
    # Reorder the matrix
    spearman_matrix = spearman_matrix.loc[corr_asc, corr_asc]

    # Create the heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(spearman_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
    plt.title(title, fontsize=16)

    ax = plt.gca()

    # Wrap x-axis tick labels to multiple lines if they are too long.
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(label.get_text(), 20))
        for label in ax.get_xticklabels()],
        rotation=90,
        fontsize=7
    )
    
    # Wrap y-axis tick labels to multiple lines if they are too long.
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(label.get_text(), 25))
        for label in ax.get_yticklabels()],
        fontsize=7
    )
    
    plt.subplots_adjust(left=0.16, bottom=0.18, top=0.92)
    plt.show()

def plot_multiple_metrics_vs_features(results_df, task, metrics=None, figsize=(14, 10)):
    """
    Plots multiple metrics into one figure with different subplots that show all models vs. number of features.
    It takes in argument the results_df generated by evaluate_k_values.
    """
    if metrics is None:
        if task == 'regression':
            metrics = ['CV_Score', 'R²_train', 'MSE_train', 'RMSE_train', 'MAE_train']
        elif task == 'classification':
            metrics = ['CV_Score', 'Accuracy_train', 'F1_macro_train', 'Precision_macro_train', 'Recall_macro_train']
        else:
            print(f"Error: Unknown task '{task}'. Please choose 'regression' or 'classification'.")
            return 
        
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    for metric in available_metrics:
        results_df[metric] = pd.to_numeric(results_df[metric], errors='coerce') # ensures that any non-numeric values present during this conversion are replaced with NaN.
    
    if PLOTS_DIR:
        os.makedirs(PLOTS_DIR, exist_ok=True)
    
    models = results_df['Model'].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        for j, model in enumerate(models):
            data = results_df[results_df['Model'] == model]
            data = data.sort_values('k')
            ax.plot(data['k'], data[metric], marker='o', label=model, color=palette[j], linewidth=2)
        
        ylabel = metric.replace('_', ' ').title()

        if metric == 'CV_Score':
            ylabel = 'CV R²-score (Validation)' if task == 'regression' else 'CV F1-Score (Validation)'
        if metric.startswith('R²'):
            ylabel = 'R²'
        elif metric.startswith('RMSE'):
            ylabel = 'RMSE'
        elif metric.startswith('MAE'):
            ylabel = 'MAE'
        elif metric.startswith('MSE'):
            ylabel = 'MSE'
        elif metric.startswith('Accuracy'):
            ylabel = 'Accuracy'
        elif metric.startswith('F1'):
            ylabel = 'F1 (macro)'
        elif metric.startswith('Precision'):
            ylabel = 'Precision (macro)'
        elif metric.startswith('Recall'):
            ylabel = 'Recall (macro)'
        
        ax.set_xlabel('Number of features (k)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        title_suffix = "" if metric == 'CV_Score' else " (train)"
        ax.set_title(f'{ylabel} {title_suffix}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(results_df['k'].unique())
        
        if i == 0:
            ax.legend() #no need to display the same legend on the 4 subplots, the first one is enough.
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'{task.capitalize()} Performance vs. Number of Features', fontsize=16)
    plt.subplots_adjust(top=0.90, hspace=0.35) 
    

    save_path = os.path.join(PLOTS_DIR, f"{task}_metrics_vs_features_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') #for better margins
    print(f"Metrics vs features plot saved to: {save_path}")
    
    plt.show()

def plot_residuals_from_all_results(all_results_regression, k_chosen, figsize=(14, 10)):
    """
    Plots residual plots for all regression models at the chosen 'k' by using the 'all_results' dictionary generated by evaluate_k_values.
    """
    data_k = all_results_regression[k_chosen]['results']
    models = list(data_k.keys())
    
    n_models = len(models)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        result = data_k[model_name]
        
        y_pred = result['y_pred']      
        y_true = result['y_test_true'] 
        residuals = y_true - y_pred
        
        # alpha=0.6 provides transparency, useful for visualizing density in overlapping points.
        ax.scatter(y_pred, residuals, alpha=0.6)
        # The horizontal line at y=0 serves as a reference, indicating where residuals should ideally fall.
        ax.axhline(y=0, color='red', linestyle='--')

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (True Value - Predicted Value)")
        ax.set_title(f"Residuals - {model_name} (k={k_chosen})")
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for j in range(n_models, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"Residual Analysis for top {k_chosen} features", fontsize=16)
    plt.tight_layout(h_pad=2.5) 
    plt.show()

    save_path = os.path.join(PLOTS_DIR, f"residual_plots_for_{k_chosen}_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') #for better margins
    print(f"residual plots for {k_chosen} features saved to: {save_path}")
    
def plot_confusion_matrices_all_models(all_results_classification, k_chosen, class_labels, figsize=(14, 10)):

    data_k = all_results_classification[k_chosen]['results']
    models = list(data_k.keys())
    
    n_models = len(models)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=12)) for label in class_labels]
    
    heatmap = None

    for i, model_name in enumerate(models):
        ax = axes[i]
        result = data_k[model_name]
        
        y_true = result['y_test_true']
        y_pred = result['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        total_samples = np.sum(cm)
        
        heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=wrapped_labels, yticklabels=wrapped_labels, ax=ax,
                    cbar=False, annot_kws={"size": 12})
        
        ax.set_title(f"{model_name} (k={k_chosen})", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        
        ax.text(1.0, -0.25, f"Total: {total_samples}", transform=ax.transAxes, 
                ha='right', va='top', fontsize=9, color='#333333')
    
    # Hide empty subplots
    for j in range(n_models, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"Confusion Matrices for top {k_chosen} features", fontsize=16)
    
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70]) 
    fig.colorbar(heatmap.collections[0], cax=cbar_ax)
    cbar_ax.set_ylabel('Count', rotation=270, labelpad=10, fontsize=10)
    
    plt.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.88, hspace=0.45, wspace=0.3)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    save_path = os.path.join(PLOTS_DIR, f"confusion_matrices_top_{k_chosen}_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to: {save_path}")

    plt.show()