-- Lists all genres in the database hbtn_0d_tvshows_rate by their rating,
-- results are sorted in descending order by their rating.
-- The database name must be passed as an argument of the mysql command.

SELECT g.name, SUM(sr.rate) AS rating
FROM tv_genres AS g
INNER JOIN tv_show_genres AS sg ON g.id = sg.genre_id
INNER JOIN tv_shows AS s ON sg.show_id = s.id
INNER JOIN tv_show_ratings AS sr ON s.id = sr.show_id
GROUP BY g.name
ORDER BY rating DESC
