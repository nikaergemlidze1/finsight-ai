"""Pydantic request schemas.

Categorical fields are restricted to the exact vocabularies the model was
trained on (see models/feature_names.json). Anything outside that set would
be one-hot encoded to all-zeros by the preprocessor (handle_unknown="ignore")
and produce a confident-looking but meaningless prediction — so we reject it
at the API boundary with a 422 instead.

Numeric bounds are generous sanity checks (reject garbage, not macro drift).
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_ex = lambda v: {"example": v}  # noqa: E731

# Vocabularies the preprocessor was fitted on.
# NOTE: the UCI dataset contains no January/February contacts and no weekend
# contacts, so the model has never seen those categories — they are excluded.
Job = Literal[
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal[
    "illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",
    "professional.course", "university.degree", "unknown",
]
YesNoUnknown = Literal["no", "yes", "unknown"]
Contact = Literal["cellular", "telephone"]
Month = Literal["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DayOfWeek = Literal["mon", "tue", "wed", "thu", "fri"]
Poutcome = Literal["failure", "nonexistent", "success"]


class CustomerInput(BaseModel):
    # Accept both alias ("emp.var.rate") and field-name ("emp_var_rate") keys
    model_config = ConfigDict(populate_by_name=True)

    # Demographics & Account
    age: int = Field(..., ge=17, le=100, json_schema_extra=_ex(35))
    job: Job = Field(..., json_schema_extra=_ex("admin."))
    marital: Marital = Field(..., json_schema_extra=_ex("married"))
    education: Education = Field(..., json_schema_extra=_ex("university.degree"))
    default: YesNoUnknown = Field(..., json_schema_extra=_ex("no"))
    housing: YesNoUnknown = Field(..., json_schema_extra=_ex("yes"))
    loan: YesNoUnknown = Field(..., json_schema_extra=_ex("no"))

    # Campaign History
    contact: Contact = Field(..., json_schema_extra=_ex("cellular"))
    month: Month = Field(..., json_schema_extra=_ex("may"))
    day_of_week: DayOfWeek = Field(..., json_schema_extra=_ex("mon"))
    campaign: int = Field(..., ge=1, le=100, json_schema_extra=_ex(1))
    # 999 = "never previously contacted" sentinel; recoded to -1 downstream
    pdays: int = Field(..., ge=-1, le=999, json_schema_extra=_ex(999))
    previous: int = Field(..., ge=0, le=100, json_schema_extra=_ex(0))
    poutcome: Poutcome = Field(..., json_schema_extra=_ex("nonexistent"))

    # Macro-Economic Indicators (loose sanity bounds only)
    emp_var_rate: float = Field(..., ge=-20, le=20,
                                alias="emp.var.rate", json_schema_extra=_ex(-1.8))
    cons_price_idx: float = Field(..., ge=50, le=150,
                                  alias="cons.price.idx", json_schema_extra=_ex(92.893))
    cons_conf_idx: float = Field(..., ge=-100, le=100,
                                 alias="cons.conf.idx", json_schema_extra=_ex(-46.2))
    euribor3m: float = Field(..., ge=-5, le=25, json_schema_extra=_ex(1.299))
    nr_employed: float = Field(..., ge=0, le=1_000_000,
                               alias="nr.employed", json_schema_extra=_ex(5099.1))


class ResearchQuery(BaseModel):
    """Body for POST /research — bounded to keep RAG/LLM cost predictable."""
    query: str = Field(..., min_length=3, max_length=2000,
                       json_schema_extra=_ex("Which customer segments convert best?"))
