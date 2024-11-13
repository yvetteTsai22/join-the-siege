# to test classifier.py
import pytest
import pandas as pd
from src.classifier import XGBoostClassifier


@pytest.fixture
def classifier():
    return XGBoostClassifier()


@pytest.mark.asyncio
async def test_predict(classifier):
    result = await classifier.predict("invoice_010.pdf")
    assert result == "invoice"
