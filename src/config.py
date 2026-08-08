TARGET = "PV1MATH"
MISSING_VALUES_THRESHOLD = 70.0
TEST_SIZE = 0.20
RANDOM_STATE = 7
KNN_NEIGHBORS = 8
MMINS_CAP = 450
VARIANCE_THRESHOLD = 0

CATEGORICAL_COLS = ["ST004D01T", "SCHLTYPE"]
ORDINAL_COLS = ["ST062Q01TA", "REPEAT", "IMMIG"]

DATA_DIR = "data"
DATASET_DIR = "dataset"

CLASS_BOUNDARIES = [float("-inf"), 420.07, 606.99, float("inf")] # Based on: OECD, Ed., PISA 2022 Technical Report. Paris: OECD Publishing, 2024.
CLASS_LABELS_DICT = {0: "Low Proficient", 1: "Medium Proficient", 2: "High Achievers"} # Based on: OECD, Ed., PISA 2022 Technical Report. Paris: OECD Publishing, 2024.
CLASS_LABELS = ["Low Proficient", "Medium Proficient", "High Achievers"]

OUTPUTS_DIR = "outputs"
SAVEDFILES_DIR = "outputs/saved_files"
TABLES_DIR = "outputs/tables"
PLOTS_DIR = "outputs/plots"
MODELS_DIR = "outputs/models"