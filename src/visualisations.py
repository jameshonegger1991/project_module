import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import textwrap
import numpy as np

def export_histograms(df: pd.DataFrame, title: str = "Histogram"):
    """
    Save histograms for all numeric columns to 'plots/univariate_plots/histograms'.

    Parameters: 
    - df: The input DataFrame containing the data.
    - title: The main title for the entire figure of histograms.
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

def export_violin_plots(df: pd.DataFrame, title: str = "Violin Plot"):
    """
    Save violin plots for all numeric columns to 'plots/univariate_plots/violin_plots/'.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - title: The main title for the entire figure of violin plots.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate violin plots.")
        return

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to generate violin plots.")
        return
    
    os.makedirs("plots/univariate_plots/violin_plots", exist_ok=True)
    
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
        
        plt.savefig(f"plots/univariate_plots/violin_plots/{col}_{title}.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"{len(numeric_cols)} violin plots saved to plots/univariate_plots/violin_plots/")

def export_barplots(df: pd.DataFrame, title: str = "Bar Plot"):

    """
    Generate bar plots for all categorical features.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - title (str): The main title for the entire figure of bar plots.
    """
    if df.empty:
        print("The DataFrame is empty. Cannot generate bar plots.")
        return

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not categorical_cols:
        print("No categorical columns found to generate bar plots.")
        return

    
    folder_path = "plots/univariate_plots/barplots"
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

def display_histograms(df: pd.DataFrame, title: str = "Histograms for all numeric features"):
    """
    Display a grid of histograms for all numeric columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The main title for the entire figure of histograms.
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

def display_violin_plots(df: pd.DataFrame, title: str = "Violin Plots - All Numeric Features"):
    """
    Generates and displays violin plots for all numeric columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - title (str): The main title for the entire figure of violin plots.
    
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

def display_barplots(df: pd.DataFrame, title: str = "Bar Plots - All Categorical Features"):
    """
    Generates and displays bar plots for all categorical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - title (str): The main title for the entire figure of bar plots.
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

def display_spearman_correlation_matrix(df: pd.DataFrame, title: str = 'Spearman Correlation Matrix'):
    """
    Displays the Spearman correlation matrix for all numeric variables.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - title (str): The title for the correlation matrix plot.
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