with source as (

    select * from {{ source('finsight_staging', 'stg_bank_marketing') }}

),

renamed as (

    select
        -- ── Customer demographics ──
        age,
        job,
        marital,
        education,

        -- ── Financial flags ──
        credit_default,
        housing_loan,
        personal_loan,

        -- ── Campaign contact ──
        contact_method,
        contact_month,
        contact_day,
        contacts_this_campaign,
        contacts_before_campaign,
        days_since_last_contact,
        was_previously_contacted,
        previous_outcome,

        -- ── Economic context ──
        emp_var_rate,
        cons_price_idx,
        cons_conf_idx,
        euribor3m,
        nr_employed,

        -- ── Target ──
        cast(subscribed as int64) as subscribed

    from source

)

select * from renamed
