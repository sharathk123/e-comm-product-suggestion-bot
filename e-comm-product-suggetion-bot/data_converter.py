import pandas as pd
from langchain_core.documents import Document

def data_converter():
    # Load data and select necessary columns
    product_data = pd.read_csv("data/flipkart_product_review.csv")
    data = product_data[['product_title', 'review']]

    # Convert DataFrame rows to a list of Document objects
    docs = [
        Document(
            page_content=row["review"],
            metadata={"product_name": row["product_title"]}
        )
        for _, row in data.iterrows()
    ]

    return docs
