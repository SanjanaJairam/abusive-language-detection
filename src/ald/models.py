"""Models: the original notebook's line-up, and the improved calibrated classifier."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC, LinearSVC

from .config import SEED
from .text import word_ngrams

try:  # The original notebook used XGBoost; fall back to scikit-learn's equivalent if it isn't installed.
    from xgboost import XGBClassifier
    def _boosting():
        return "XGBoost", XGBClassifier(random_state=SEED)
except ImportError:
    def _boosting():
        return "Gradient boosting (sklearn; XGBoost unavailable)", HistGradientBoostingClassifier(random_state=SEED)


def original_models():
    """The supervised models from the notebook, on averaged GloVe-25 features, same settings."""
    boost_name, boost = _boosting()
    return {
        "SVM (rbf, C=10, tuned)": SVC(C=10, kernel="rbf", random_state=SEED),
        "SVM (default)": SVC(random_state=SEED),
        "Random forest (depth 2)": RandomForestClassifier(max_depth=2, random_state=0),
        boost_name: boost,
        "AdaBoost (2 estimators)": AdaBoostClassifier(n_estimators=2, random_state=0),
    }


def bigram_svm():
    """Bigram TF-IDF + linear SVM: the best combination in the comparative study our 2023
    literature survey planned to build on (Ranka, Jairam & Sreekanth, 2023)."""
    return Pipeline([("features", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", LinearSVC(random_state=SEED))])


def balanced_glove_svm():
    """Same features and SVM as the best original model, but weighting the rare hate class up."""
    return SVC(C=10, kernel="rbf", class_weight="balanced", random_state=SEED)


def improved_model(negation=False):
    """TF-IDF word (1-2) and character (2-5) n-grams with class-balanced logistic regression,
    following the feature design of Davidson et al. (2017).

    Character n-grams help with spelling variation and obfuscation; balanced class weights
    stop the 6% hate class from being ignored. With `negation=True`, word n-grams are built
    from negation-marked tokens (see text.negation_tokens). Probabilities are calibrated separately.
    """
    words = (TfidfVectorizer(analyzer=word_ngrams, min_df=2, sublinear_tf=True) if negation
             else TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True))
    features = FeatureUnion([
        ("words", words),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000)
    return Pipeline([("features", features), ("clf", clf)])


def calibrate(fitted_model, X_val, y_val):
    """Sigmoid calibration on held-out data, so a 0.8 hate score means roughly 80% of such posts are hate."""
    cal = CalibratedClassifierCV(FrozenEstimator(fitted_model), method="sigmoid")
    return cal.fit(X_val, y_val)
