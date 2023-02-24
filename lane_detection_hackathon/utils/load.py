from datetime import date


def get_date_string() -> str:
    return date.today().strftime("%Y-%m-%d")
