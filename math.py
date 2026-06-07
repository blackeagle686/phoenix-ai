def sum(a, b):
    """Return the sum of two numbers."""
    return a + b

def subtract(a, b):
    """Return the difference of two numbers (a - b)."""
    return a - b

def multiply(a, b):
    """Return the product of two numbers."""
    return a * b

def divide(a, b):
    """Return the quotient of a divided by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
