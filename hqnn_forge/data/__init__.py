"""
hqnn_forge.data
===============
Dataset loaders.

Exported symbols
----------------
load_credit_card_fraud   Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud).
CreditCardFraud          (X, y, feature_names) returned by the loader.
DatasetNotFoundError     Raised when the CSV is absent and download=False.
DatasetDownloadError     Raised when download=True but fetching the CSV fails.
load_taiwanese_bankruptcy  UCI Taiwanese Bankruptcy Prediction (3.2% positive).
load_iranian_churn         UCI Iranian Churn (15.7% positive).
load_cervical_cancer_risk  UCI Cervical Cancer (Risk Factors), biopsy label (6.7% positive).
BinaryDataset            (X, y, feature_names) returned by the UCI loaders.
"""

from hqnn_forge.data.credit_card import (
    CreditCardFraud,
    DatasetDownloadError,
    DatasetNotFoundError,
    load_credit_card_fraud,
)
from hqnn_forge.data.uci import (
    BinaryDataset,
    load_cervical_cancer_risk,
    load_iranian_churn,
    load_taiwanese_bankruptcy,
)

__all__: list[str] = [
    "BinaryDataset",
    "CreditCardFraud",
    "DatasetDownloadError",
    "DatasetNotFoundError",
    "load_cervical_cancer_risk",
    "load_credit_card_fraud",
    "load_iranian_churn",
    "load_taiwanese_bankruptcy",
]
