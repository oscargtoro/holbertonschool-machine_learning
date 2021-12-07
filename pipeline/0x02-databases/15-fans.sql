-- Ranks country origins of bands, ordered by the number of (non-unique) fans.
-- Uses the metal_bands.sql dump

SELECT origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC
