

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


DATA_PATH = RAW_DIR / "clickbait_dataset.csv"
TEXT_COL = "title"
LABEL_COL = "clickbait"
POSITIVE_LABELS = {"clickbait", "1", 1, True}


RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10  # carved out of the train portion

TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True


VOCAB_SIZE = 20_000   # top-N most frequent tokens kept
MAX_LEN = 30          # words per headline after padding/truncation
EMBED_DIM = 64        # embedding dimension learned from scratch
PAD_IDX = 0
UNK_IDX = 1


BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 4          # early-stopping patience on val loss
GRAD_CLIP = 5.0
DEFAULT_THRESHOLD = 0.5

SHALLOW_HIDDEN = (128,)
SHALLOW_DROPOUT = 0.30

DEEP_HIDDEN = (256, 128, 64)
DEEP_DROPOUT = 0.40   # heavier regularisation for the deeper net

CNN_FILTER_SIZES = (3, 4, 5)
CNN_NUM_FILTERS = 64
CNN_DROPOUT = 0.50
