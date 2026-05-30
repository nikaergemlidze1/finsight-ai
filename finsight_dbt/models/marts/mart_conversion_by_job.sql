with contacts as (

    select * from {{ ref('stg_bank_marketing') }}

),

aggregated as (

    select
        job,
        count(*)                              as total_contacts,
        sum(subscribed)                       as total_subscribers,
        round(avg(subscribed) * 100, 1)       as conversion_pct
    from contacts
    group by job

)

select *
from aggregated
order by conversion_pct desc
