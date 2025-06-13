
from pydantic import BaseModel
from typing import Optional

class ProductInfo(BaseModel):
    product_name: str
    product_details: str
    tentative_price_usd: Optional[float] = None