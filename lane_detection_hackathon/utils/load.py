from datetime import date

from .types import Dict


def get_date_string() -> str:
    return date.today().strftime("%Y-%m-%d")


def get_label_map() -> Dict:
    return Dict()
