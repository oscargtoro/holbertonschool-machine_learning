-- Computes the score average of all records in the table second_table in
-- your MySQL server.
-- The result column will be named average.
-- The database name must be passed as an argument of the mysql command.

SELECT TRUNCATE(AVG(score), 2) AS average FROM second_table;
