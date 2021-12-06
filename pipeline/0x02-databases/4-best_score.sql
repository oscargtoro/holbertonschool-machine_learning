-- Lists all records with a score >= 10 in the table second_table in your
-- MySQL server.
-- Results display both the score and the name (in this order).
-- Records are be ordered by score (top first).
-- The database name must be passed as an argument of the mysql command.

SELECT score, name FROM second_table WHERE score >= 10 ORDER BY score DESC;
