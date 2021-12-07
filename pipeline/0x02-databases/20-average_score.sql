-- Procedure that computes and stores the average score for a student.
-- Uses the DB structure in 20-init.sql.

DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id INT)
BEGIN
    SET @scr_avg = (SELECT AVG(c.score)
                    FROM corrections AS c
                    WHERE c.user_id = user_id);
    UPDATE users SET average_score = @scr_avg where id = user_id;
END//
DELIMITER ;
