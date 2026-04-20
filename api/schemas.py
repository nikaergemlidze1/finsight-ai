from pydantic import BaseModel, Field

_ex = lambda v: {"example": v}  # noqa: E731


class CustomerInput(BaseModel):
    # Demographics & Account
    age: int = Field(..., json_schema_extra=_ex(35))
    job: str = Field(..., json_schema_extra=_ex("admin."))
    marital: str = Field(..., json_schema_extra=_ex("married"))
    education: str = Field(..., json_schema_extra=_ex("university.degree"))
    default: str = Field(..., json_schema_extra=_ex("no"))
    housing: str = Field(..., json_schema_extra=_ex("yes"))
    loan: str = Field(..., json_schema_extra=_ex("no"))

    # Campaign History
    contact: str = Field(..., json_schema_extra=_ex("cellular"))
    month: str = Field(..., json_schema_extra=_ex("may"))
    day_of_week: str = Field(..., json_schema_extra=_ex("mon"))
    campaign: int = Field(..., json_schema_extra=_ex(1))
    pdays: int = Field(..., json_schema_extra=_ex(999))
    previous: int = Field(..., json_schema_extra=_ex(0))
    poutcome: str = Field(..., json_schema_extra=_ex("nonexistent"))

    # Macro-Economic Indicators
    emp_var_rate: float = Field(..., alias="emp.var.rate", json_schema_extra=_ex(-1.8))
    cons_price_idx: float = Field(..., alias="cons.price.idx", json_schema_extra=_ex(92.893))
    cons_conf_idx: float = Field(..., alias="cons.conf.idx", json_schema_extra=_ex(-46.2))
    euribor3m: float = Field(..., json_schema_extra=_ex(1.299))
    nr_employed: float = Field(..., alias="nr.employed", json_schema_extra=_ex(5099.1))
