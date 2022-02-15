SELECT
    ClientID,
    Amount,
    OpeartionTime,
    row_number() over (
        PARTITION by ClientID, SignOfPayment
        ORDER BY
            OpeartionTime
    ) as rn,
 lead(OpeartionTime) as NextBuy