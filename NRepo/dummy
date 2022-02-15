SELECT
    date_part('month', transaction_date :: date) AS MONTH,
    action,
    servtype,
    is_partner,
    count(user_id) AS user_cnt,
    count(transaction_id AS transaction_cnt),
    count(service_id) AS service_cnt,
    sum(payment) AS sum_payment,
    sum(primecost) AS sum_primecost
FROM
    transactions
GROUP BY
    date_part('month', transaction_date :: date)
    action,
    servtype,
    is_partner
ORDER BY
    date_part('month', transaction_date :: date)