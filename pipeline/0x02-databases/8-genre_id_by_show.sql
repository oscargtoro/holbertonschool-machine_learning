-- Lists all shows contained in hbtn_0d_tvshows that have at least one genre
-- linked sorted in ascending order by tv_shows.title and
-- tv_show_genres.genre_id.
-- The database name must be passed as an argument of the mysql command.

SELECT s.title, g.genre_id
FROM tv_shows AS s, tv_show_genres AS g
WHERE s.id = g.show_id
ORDER BY s.title ASC, g.genre_id ASC;
