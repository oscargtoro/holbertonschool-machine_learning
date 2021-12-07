-- Resets the attribute valid_email only when the email has been changed.
-- Requires to first run 18-init.sql, 18-main.sql can be used for testing

CREATE TRIGGER email_reset BEFORE UPDATE ON users
FOR EACH ROW
    SET NEW.valid_email = IF(NEW.email <> OLD.email, 0, NEW.valid_email)
