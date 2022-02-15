1. Во-первых, необходимо убрать подзапрос

```sql {.line-numbers}
select 
case when current_date-month_paid_inst<'10 month' then 1 
when current_date-month_paid_inst<'25 month' then 2 
when current_date-month_paid_inst<'50 month' then 2 
when current_date-month_paid_inst<'200 month' then 3 else -1 end len_bin
,count(*) as cnt
,region_bin
,sum(amt_credit)
,avg(amt_credit)
from (
select * from (
select    
id_client, 
date_trunc('month',last_paid_inst) month_paid_inst,
case when name_region = 'Центр' then 1 
when name_region = 'Север' then 2 else -1 end region_bin, 
amt_credit
sum(amt_credit) over (partition by name_region order by last_paid_inst) as sum_all_by_region
from skybank.early_collection_clients a 
    left join skybank.region_dict b 
        on a.id_city = b.id_city
where lower(name_region) not like 'сибирь' or upper(name_region) not like 'СИБИРЬ' union
select 
    id_client 
    ,'1900-01-01' 
    ,month_paid_inst
    ,case when name_region = 'Центр' then 1 when name_region = 'Север' then 2 else -1 end 
    ,amt_loan
    ,sum(amt_loan) over (partition by name_region) as sum_all_by_region
from skybank.late_collection_clients a 
    join skybank.region_dict b on 
        a.id_city = b.id_city 
    where name_region not like '%сибирь%') tt
group by len_bin, region_bin
```
