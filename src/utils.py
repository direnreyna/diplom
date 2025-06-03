# src/utils.py

def int_yyyymmdd(date_str: str) -> int:
    """Преобразует 'dd.mm.yyyy' → целое число"""
    d, m, y = date_str.split('.')
    return int(f"{y}{m}{d}")

def first_date_is_newer(date_str1: str, date_str2: str) -> bool:
    """Возвращает True, если первая дата новее второй"""
    return int_yyyymmdd(date_str1) > int_yyyymmdd(date_str2)