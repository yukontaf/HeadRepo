SELECT
    country,
    event_type,
    cnt,
    cnt_installs,
    cnt_trials,
    cnt_purchases,
    cnt_installs / cnt_purchases AS conversion
FROM
    (
        SELECT
            country,
            event_type,
            count(event_type) AS cnt,
            sum(
                CASE
                    WHEN event_type = 'install' THEN 1
                    ELSE 0
                END
            ) :: float AS cnt_installs,
            sum(
                CASE
                    WHEN event_type = 'trial' THEN 1
                    ELSE 0
                END
            ) :: float AS cnt_trials,
            sum(
                CASE
                    WHEN event_type = 'purchase' THEN 1
                    ELSE 0
                END
            ) :: float AS cnt_purchases
        FROM
            EVENTS
        GROUP BY
            event_type,
            country
    )