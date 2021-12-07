-- Trigger that decreases the quantity of an item after adding a new order.
-- Requires to first run 17-init.sql, 17-main.sql can be used for testing

CREATE TRIGGER items_sub BEFORE INSERT ON orders
    FOR EACH ROW
    UPDATE items SET quantity = quantity - NEW.number
    WHERE name = NEW.item_name
