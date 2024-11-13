# to test classifier.py
import pytest
import pandas as pd
from src.classifier import XGBoostClassifier, XGBoostClassifierSemantic


@pytest.fixture
def classifier():
    # return XGBoostClassifier()
    return XGBoostClassifierSemantic()


@pytest.mark.asyncio
async def test_predict(classifier):
    result = await classifier.predict("invoice_010.pdf")
    assert result == "invoice"


@pytest.mark.asyncio
async def test_predict2(classifier):
    result = await classifier.predict("bank_statement_1.pdf")
    assert result == "bank_statement"


@pytest.mark.asyncio
async def test_predict3(classifier):
    result = await classifier.predict("drivers_license_2.pdf")
    assert result == "driver_license"
