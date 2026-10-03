from pydantic import BaseModel, ConfigDict, Field, model_validator
from telecom_support.taxonomy import Category, Product, Severity, Sentiment

class ComplaintLabels(BaseModel):
    model_config = ConfigDict(extra='forbid')
    categories: list[Category] = Field(min_length=1, max_length=4)
    products: list[Product] = Field(min_length=1, max_length=3)
    severity: Severity
    sentiment: Sentiment

    @model_validator(mode='after')
    def unique_labels(self):
        if len(set(self.categories)) != len(self.categories) or len(set(self.products)) != len(self.products):
            raise ValueError('Duplicate labels')
        return self
