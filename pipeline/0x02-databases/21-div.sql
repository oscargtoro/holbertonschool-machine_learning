-- Creates a function that divides (and returns) the first by the second
-- number or returns 0 if the second number is equal to 0.
-- Uses the DB structure in 21-init.sql.


CREATE FUNCTION SafeDiv (a DOUBLE PRECISION, b DOUBLE PRECISION)
RETURNS FLOAT DETERMINISTIC
RETURN IF(b <> 0, a / b, 0);
