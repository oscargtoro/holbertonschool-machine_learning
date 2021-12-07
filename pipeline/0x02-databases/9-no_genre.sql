-- Lists all shows contained in hbtn_0d_tvshows without a genre linked
-- results are sorted in ascending order by tv_shows.title and tv_show_genres.genre_id.
-- The database name must be passed as an argument of the mysql command.

SELECT s.title, g.genre_id
FROM tv_shows AS s
LEFT JOIN tv_show_genres AS g ON s.id = g.show_id
WHERE g.genre_id IS NULL
ORDER BY s.title ASC, g.genre_id ASC;
