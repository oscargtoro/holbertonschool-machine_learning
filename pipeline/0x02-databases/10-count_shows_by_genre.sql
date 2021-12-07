-- Lists all genres from hbtn_0d_tvshows and displays the number of shows
-- linked to each, genres with no shows linked aren't displayed, results
-- are sorted in descending order by the number of shows linked.
-- The database name must be passed as an argument of the mysql command.

SELECT g.name as genre, COUNT(sg.show_id) as number_of_shows
FROM tv_genres AS g
LEFT JOIN tv_show_genres AS sg ON g.id = sg.genre_id
GROUP BY g.name
ORDER BY number_of_shows DESC
