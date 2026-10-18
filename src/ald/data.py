"""Loading the datasets used in this project. See get_data.sh for where each one comes from."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DAVIDSON_CSV, HATECHECK_CSV, MADLIBS_CSV, SEED


def load_davidson():
    """Davidson et al. (2017): 24,783 tweets labelled hate (0), offensive (1) or neither (2)."""
    df = pd.read_csv(DAVIDSON_CSV)
    return df[["tweet", "class"]].rename(columns={"class": "label"}).reset_index(drop=True)


def split(df):
    """Stratified 60/20/20 train/validation/test split with a fixed seed.

    Validation is used only to calibrate probabilities and choose policy thresholds.
    Model and policy results come from the untouched test set; the dialect audit, which needs
    more examples per group, uses 5-fold cross-validated predictions instead (see run_all.py).
    """
    train, rest = train_test_split(df, test_size=0.4, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=SEED)
    # Keep each tweet's original row number in an "index" column.
    return train.reset_index(), val.reset_index(), test.reset_index()


def load_hatecheck():
    """Röttger et al. (2021): 3,728 functional test cases across 29 functionalities."""
    df = pd.read_csv(HATECHECK_CSV)
    return df[["functionality", "test_case", "label_gold", "target_ident"]]


def load_madlibs():
    """Dixon et al. (2018): 76,564 templated sentences with identity terms, labelled BAD / NOT_BAD."""
    return pd.read_csv(MADLIBS_CSV)
