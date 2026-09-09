"""Intentional logic-error fixture for the LogicFlow repair workflow."""


def calculate_total(price, quantity):
    # BUG: this should multiply price by quantity.
    return price + quantity