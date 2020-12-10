# 0x02. Calculus

## Concepts Learned

- Summation and Product notation
- What is a series?
- Common series
- What is a derivative?
- What is the product rule?
- What is the chain rule?
- Common derivative rules
- What is a partial derivative?
- What is an indefinite integral?
- What is a definite integral?
- What is a double integral?

## Files 1-8 and 11-16

Answers to general calculus questions.

## 9-sum_total.py

Calculates the sum of **_i to the power of 2 from i equals 1 to n_**.

```
>>> summation = __import__('9-sum_total').summation_i_squared
>>>
>>> n = 5
>>> print(summation(n))
55
>>>
```

## 10-matisse.py

Calculates the derivate of a polynomial:

- Recieves a list of coefficients representing a polynomial.
  - the index of the list represents the power of **x** that the coefficient belongs to.
  - Example: if **_f(x) = x^3 + 3x +5_**, the list is equal to **[5, 3, 0, 1]**.
- Returns 0 when the list is not valid.
- If the derivative is **0**, returns **[0]**.
- Returns a new list of coefficients representing the derivative of the polynomial.

```
>>> derivative = __import__('10-matisse').poly_derivative
>>> poly = [5, 3, 0, 1]
>>> print(derivative(poly))
[3, 0, 3]
>>>
```
