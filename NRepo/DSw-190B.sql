WITH row_n AS (
    SELECT t.*,
           row_number() over (
               PARTITION by date_trunc('month', t.date)
               ORDER BY
                   t.amount DESC
               )                                                       AS rn,
           round(
                       0.05 * (
                       count(*) over (PARTITION by date_trunc('month', t.date))
                       )
               )                                                       AS cnt,
           date_trunc('month', t.date)                                 AS MONTH,
           sum(amount) over (PARTITION by date_trunc('month', t.date)) AS month_sum
    FROM semrush_bank t
)
SELECT tt.id,
       tt.date,
       tt.amount,
       tt.month_sum
FROM row_n tt
WHERE rn <= cnt