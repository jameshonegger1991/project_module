from IPython import get_ipython
from IPython.display import Image, display
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import os

def print_files(file_path: str):
    with open(file_path, 'r') as file:
        content_to_read = file.read()
        print(content_to_read)

def check_overfitting_regression(train_r2, cv_score):

    # Overfitting check based on check based on the gap between Train R² and Cross-Validation R². 
    # Inspiration for the heuristic: https://datascience.stackexchange.com/questions/77298/how-many-ways-are-there-to-check-model-overfitting)
    
    gap = train_r2 - cv_score
    
    # # A 10% R² gap is a clear overfitting signal, 5% can be considered as a warning zone whiles negative gaps below -5% are rare enough to be noted as positive.
    if gap > 0.10:
        status = "Overfitting"
        detail = f"Gap (Train - CV) = {gap:.4f} (> 0.10)"
    elif gap > 0.05:
        status = "Mild overfitting"
        detail = f"Gap (Train - CV) = {gap:.4f}"
    elif gap < -0.05:
        status = "Good generalisation (Test R² > Train R²)"
        detail = f"Gap (Train - CV) = {gap:.4f}"
    else:
        status = "Good generalisation"
        detail = f"Gap (Train - CV) = {gap:.4f}"
    
    return status, detail, gap

def check_overfitting_classification(train_f1, cv_f1):

    # Overfitting check based on the gap between Train F1 and Cross-Validation F1.
    # In that situation, the threshold is stricter (0.05) than regression (0.10) because classification 
    # metrics are strictly bounded between 0 and 1. This means that in such situation, a 5% drop 
    # represents a critical loss of operational predictive power.
    gap = train_f1 - cv_f1
    
    if gap > 0.05:
        status = "Overfitting"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    elif gap > 0.02:
        status = "Mild overfitting"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    elif gap < -0.03: # A negative gap exceeding 3% is relatively rare in practice and suggests that the model generalises surprisingly well.
        status = "Good generalisation (CV > Train)"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    else:
        status = "Good generalisation"
        detail = f"F1 Macro Gap (Train - CV) = {gap:.4f}"
    
    return status, detail, gap

def display_visualisation(path):
    """
    Display a saved PNG image in both Jupyter Notebooks and standard Python scripts.
    """
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return

    if get_ipython() is not None: #for jupyternotebook
        display(Image(filename=path))

    else: # for standard python script
        img = PILImage.open(path)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        ax.axis("off")

        plt.show(block=True)
        #print("Press any key or click on the figure to continue...")
        #plt.waitforbuttonpress()
        #plt.close(fig)