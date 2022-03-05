WITH row_n AS (
    SELECT t.*,
           date_trunc('month', t.date)                                 as payment_month,
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
    FROM payments t
)
SELECT tt.id,
       tt.payment_month,
       tt.date,
       tt.amount,
       tt.month_sum
FROM row_n tt
WHERE rn <= cnt