-- Lists all shows from hbtn_0d_tvshows_rate by their rating, sorted in
-- descending order by rating.
-- The database name must be passed as an argument of the mysql command.

SELECT s.title, SUM(sr.rate) AS rating
FROM tv_shows AS s
JOIN tv_show_ratings AS sr ON s.id = sr.show_id
GROUP BY s.title
ORDER BY rating DESC
