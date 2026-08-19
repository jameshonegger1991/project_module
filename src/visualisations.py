import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import textwrap
import numpy as np
import shap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder
from src.config import (
    PLOTS_DIR,
)

# GLOBAL EXPLORATORY DATA ANALYSIS (ON WHOLE SET)

def save_individual_histograms(df, title = "Histogram"):
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
    
    os.makedirs(f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/histograms", exist_ok=True)
    
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
        plt.savefig(f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/histograms/{col}_{title}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n • {len(numeric_cols)} histograms saved to {PLOTS_DIR}/EDA/whole_dataset/univariate_plots/histograms")

def save_individual_violin_plots(df, title = "Violin Plot"):
    """
    Save violin plots for all numeric columns to 'outputs/plots/EDA/whole_dataset/univariate_plots/violin_plots/'.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate violin plots.")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to generate violin plots.")
        return
    
    os.makedirs(f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/violin_plots", exist_ok=True)
    
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
        
        plt.savefig(f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/violin_plots/{col}_{title}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n • {len(numeric_cols)} violin plots saved to {PLOTS_DIR}/EDA/whole_dataset/univariate_plots/violin_plots")

def save_individual_barplots(df, title = "Bar Plot"):
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

    folder_path = f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/barplots" 
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
    
    print(f"\n • {len(categorical_cols)} bar plots saved to {folder_path}/")

def save_combined_histograms(df, title = "Histograms - All Numeric Features (Before Cleaning And Imputation)"):
    """
    Save a combined grid of histograms for all numeric columns.
    """ 
    if df.empty:
        print("The DataFrame is empty. Cannot generate histograms.")
        return

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to display histograms.")
        return

    folder_path = f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/histograms"
    os.makedirs(folder_path, exist_ok=True)

    axes = df[numeric_cols].hist(figsize=(22, 18), bins=30, xlabelsize=8, ylabelsize=8)

    for ax in axes.ravel():
        ax.set_title(ax.get_title(), fontsize=8)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.5)
    plt.subplots_adjust(hspace=0.9, wspace=0.4)
    plt.savefig(f"{folder_path}/{title}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n • {title}.png saved to {folder_path}/")

def save_combined_violin_plots(df, title = "Violin Plots - All Numeric Features (Before Cleaning And Imputation)"):
    """
    Save a combined grid of violin plots for all numeric columns. 
    Inspiration: https://www.datasciencebyexample.com/2022/05/15/2022-05-15-1/ 
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate violin plots.")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to generate violin plots.")
        return

    folder_path = f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/violin_plots"
    os.makedirs(folder_path, exist_ok=True)

    n_cols = 4 

    # Source: https://discuss.python.org/t/integer-ceiling-divide/91269 
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols 
    
    # Source: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4.5)) # Adjusted figsize for better layout

    # Flatten subplot axes for iteration.
    # Source: - https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    #         - https://stackoverflow.com/questions/62035244/creating-multiple-plot-using-for-loop-from-dataframe 
    axes = axes.flatten()
    
    for i in range(n_rows):
        for j in range(n_cols):
            index = i * n_cols + j 
            if index < len(numeric_cols):
                col = numeric_cols[index]
                sns.violinplot(data=df[col], orient='h', ax=axes[index]) 
                axes[index].set_title(col, fontsize=8) 
                axes[index].set_xlabel('') 
                axes[index].tick_params(labelsize=8) 
            else:
                # Hide unused axes.
                axes[index].axis('off')
    
    plt.suptitle(title, fontsize=16) 
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=2.5) 
    plt.subplots_adjust(hspace=1.0, wspace=0.4)
    plt.savefig(f"{folder_path}/{title}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n • {title}.png saved to {folder_path}/")
    
def save_combined_barplots(df, title = "Bar Plots - All Categorical Features (Before Cleaning And Imputation)"):
    """
    Save a combined grid of bar plots for all categorical columns in the DataFrame.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate bar plots.")
        return

    #This function is based on the same model as "display_violin_plots"
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not cat_cols:
        print("No categorical columns found to generate bar plots.")
        return

    n_cols = 2

    folder_path = f"{PLOTS_DIR}/EDA/whole_dataset/univariate_plots/barplots"
    os.makedirs(folder_path, exist_ok=True)

    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = np.asarray(axes).flatten() 
    
    for i in range(len(cat_cols)):
        col = cat_cols[i]
        counts = df[col].value_counts()

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
    plt.savefig(f"{folder_path}/{title}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n • {title}.png saved to {folder_path}/")

# GLOBAL EXPLORATORY DATA ANALYSIS (ON TRAIN SET)
def save_spearman_correlation_matrix(df, title = 'Spearman Correlation Matrix'):
    """
    Generates the Spearman correlation matrix for all numeric variables.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate Spearman correlation matrix.")
        return

    folder_path = f"{PLOTS_DIR }/EDA/train_set/multivariate_plots"
    os.makedirs(folder_path, exist_ok=True)

    spearman_matrix = df.corr(method='spearman')
    corr_asc = spearman_matrix["PV1MATH"].abs().sort_values(ascending=False).index
    spearman_matrix = spearman_matrix.loc[corr_asc, corr_asc]

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
    
    # Same for y-axis labels
    ax.set_yticklabels(
        ["\n".join(textwrap.wrap(label.get_text(), 25))
        for label in ax.get_yticklabels()],
        fontsize=7
    )
    
    plt.subplots_adjust(left=0.16, bottom=0.18, top=0.92)
    plt.savefig(f"{folder_path}/{title}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n • {title}.png saved to {folder_path}/")

# MODEL TRAINING
def save_multiple_metrics_vs_features_plot(results_df, task, metrics=None, figsize=(14, 10)):
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
        os.makedirs(f"{PLOTS_DIR}/model_training", exist_ok=True)
    
    models = results_df['Model'].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))
    
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # INSPIRATION: https://stackoverflow.com/questions/66705955/creating-subplots-through-a-loop-from-dataframe 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]

        #INSPIRATION: https://stackoverflow.com/questions/26355313/plotting-multiple-plots-generated-inside-a-for-loop-on-the-same-axess 
        for j, model in enumerate(models):
            data = results_df[results_df['Model'] == model]
            data = data.sort_values('k')
            ax.plot(data['k'], data[metric], marker='o', label=model, color=palette[j], linewidth=2)
        
        ylabel = metric.replace('_', ' ').title()

        if metric == 'CV_Score':
            ylabel = 'CV R²-score (Validation)' if task == 'regression' else 'CV F1 (macro) - Score (Validation)'
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
        ax.legend() 
    
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(f'{task.capitalize()} Performance vs. Number of Features', fontsize=16, y=0.96)
    plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.45, wspace=0.30)

    save_path = os.path.join(PLOTS_DIR,"model_training",f"{task}_metrics_vs_features_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') #for better margins
    plt.close()
    print(f"\n • Metrics vs features plot saved to: {save_path}")

def save_residuals_plot_from_all_results(all_results_regression, k_chosen, figsize=(14, 10)):
    """
    Plots residual plots for all regression models at the chosen 'k' by using the 'all_results' dictionary generated by evaluate_k_values.
    (Custom wrapper designed for pipeline consistency over sklearn's PredictionErrorDisplay).
    """
    os.makedirs(f"{PLOTS_DIR}/model_training", exist_ok=True)
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
        

        ax.scatter(y_pred, residuals, alpha=0.6)
        # y=0 serves as a reference
        ax.axhline(y=0, color='red', linestyle='--')

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (True Value - Predicted Value)")
        ax.set_title(f"Residuals - {model_name} (k={k_chosen})")
        ax.grid(True, alpha=0.3)
    
    for j in range(n_models, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"Residual Analysis for top {k_chosen} features", fontsize=16)
    plt.tight_layout(h_pad=2.5) 

    save_path = os.path.join(PLOTS_DIR,"model_training",f"regression_residual_plots_for_{k_chosen}_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') #for better margins
    plt.close()
    print(f"\n • regression_residual plots for {k_chosen} features saved to: {save_path}")
    
def save_confusion_matrices_plot_all_models(all_results_classification, k_chosen, class_labels, figsize=(14, 10)):
    """
    plot confusion matrices for all classification models built with k-top features.
    (Custom wrapper designed for ensuring pipeline consistency and custom metadata injection over sklearn's ConfusionMatrixDisplay).

    INSPIRATIONS: - https://stackoverflow.com/questions/28356359/one-colorbar-for-seaborn-heatmaps-in-subplot 
                  - https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots 
                  - https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side 
    """
    data_k = all_results_classification[k_chosen]['results']
    models = list(data_k.keys())
    
    n_models = len(models)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=12)) for label in class_labels] # required, otherwise overlapping on x/y axes
    
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
    
    for j in range(n_models, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"Confusion Matrices for top {k_chosen} features", fontsize=16)

    # A unified global colorbar is preferred to avoid redundant legends across subplots
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.70]) 
    fig.colorbar(heatmap.collections[0], cax=cbar_ax)
    cbar_ax.set_ylabel('Count', rotation=270, labelpad=10, fontsize=10)

    # Withouth this fine-tune layout margins, titles and labels clip
    plt.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.88, hspace=0.45, wspace=0.3)

    os.makedirs(f"{PLOTS_DIR}/model_training", exist_ok=True)
    save_path = os.path.join(PLOTS_DIR,"model_training",f"confusion_matrices_top_{k_chosen}_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n • Confusion matrices saved to: {save_path}")

def save_learning_curves_plot_all_models(all_results, k_chosen, X_train_after_var_thresh, y_train, task='regression', figsize=(14, 10)):
    """
    plot learning curves for all models built with k-top features.
    REFERENCE: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html 
    INSPIRATION: https://stackoverflow.com/questions/41097322/learning-curve-high-bias-high-variance-why-the-testing-learning-curve-gets-f
    """
    data_k = all_results[k_chosen]['results']
    models = list(data_k.keys())
    
    n_models = len(models)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    indices = all_results[k_chosen]['indices']
    X_train_k = X_train_after_var_thresh[:, indices]
    scoring = 'f1_macro' if task == 'classification' else 'r2'
    metric_label = "F1 (macro)" if task == 'classification' else "R² Score"
    
    for i, model_name in enumerate(models):
        ax = axes[i]
        result = data_k[model_name]
        model_entry = result['model']
        estimator = model_entry.best_estimator_ if hasattr(model_entry, 'best_estimator_') else model_entry

        y_train_use = y_train
        if model_name == "XGBoost" and task == 'classification':
            encoder = LabelEncoder()
            y_train_use = encoder.fit_transform(y_train)

        train_sizes, train_scores, test_scores = learning_curve(
            estimator=estimator,
            X = X_train_k,
            y = y_train_use,
            train_sizes = np.linspace(0.1, 1.0, 5), #  to evaluate behavior with 10%, 32.5%, 55%, 77.5% and 100% of training data.
            cv = 3, 
            scoring=scoring,
            n_jobs=-1,
            random_state=7
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        
        ax.set_title(f"Learning Curve - {model_name} (k={k_chosen})", fontsize=12)
        ax.set_xlabel("Training examples size", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    
    for j in range(n_models, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"Learning Curves for all models (top {k_chosen} features)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(f"{PLOTS_DIR}/model_training", exist_ok=True)
    save_path = os.path.join(PLOTS_DIR,"model_training",f"{task}_learning_curves_top_{k_chosen}_features.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n • Learning curves saved to: {save_path}")

# SHAP ANALYSIS 
def barplot_global_shap_rankings(global_mean_absolute_shap_rankings_dict, top_k=10):

    n_models = len(global_mean_absolute_shap_rankings_dict)
    if n_models == 0:
        print("No models found in the dictionary to plot.")
        return

    ncols = 2 if n_models > 1 else 1
    nrows = (n_models + 1) // 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, (model_name, shap_ranking_df) in enumerate(global_mean_absolute_shap_rankings_dict.items()):
        ax = axes[i]
        df_plot = shap_ranking_df.head(top_k).copy().iloc[::-1]
        
        ax.barh(df_plot['Feature'], df_plot['Mean_Abs_SHAP'], color='blue', edgecolor='black', alpha=0.85)
        
        ax.set_xlabel('Mean Absolute SHAP Value', fontsize=10, fontweight='bold')
        ax.set_ylabel('Features', fontsize=10, fontweight='bold')
        ax.set_title(f'Top {top_k} SHAP Importance - {model_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    for j in range(n_models, len(axes)):
        axes[j].axis('off')
    fig.suptitle("Global SHAP Feature Importances", fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs(f"{PLOTS_DIR}/SHAP_analysis", exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "SHAP_analysis", f"Global_SHAP_feature_importances_barplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') #for better margins
    plt.close()
    print(f"\n • Global SHAP feature importances barplot saved to: {save_path}")

def shap_summary_plot(shap_values, X_test, feature_names):

    #REFERENCE: https://medium.com/womenintechnology/understanding-model-predictions-with-shap-d7457f6a31c3
    n_models = len(shap_values)
    if n_models == 0:
        print("No models found in the dictionary to plot.")
        return

    ncols = 2 if n_models > 1 else 1
    nrows = (n_models + 1) // 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, (model_name, model_data) in enumerate(shap_values.items()):
        ax = axes[i]
        
        shap_vals = model_data['shap_values']
        x_subset = model_data['X_test_subset']

        explanation = shap.Explanation(
            values = shap_vals,
            data = x_subset,
            feature_names = feature_names
        )
        
        shap.plots.beeswarm(explanation, ax=ax, plot_size = None, show = False)
        
        ax.set_xlabel('SHAP Value (Impact on model output)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Features', fontsize=10, fontweight='bold')
        ax.set_title(f'SHAP Importance - {model_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    for j in range(n_models, len(axes)):
        axes[j].axis('off')

    fig.suptitle("SHAP Summary plot", fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs(f"{PLOTS_DIR}/SHAP_analysis", exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, "SHAP_analysis", f"SHAP_summary_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()
    print(f"\n • SHAP summary plot saved to: {save_path}")
    

