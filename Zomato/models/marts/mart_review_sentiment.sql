{{ config(
    materialized='table',
    tags=['ai']
) }}

with enriched_reviews as (
    select
        review_id,
        order_id,
        user_id as customer_id,
        restaurant_id::string as restaurant_id,
        rating,
        comment,
        review_date,
        sentiment,
        sentiment_score,
        primary_aspect,
        tags,
        key_issue_or_highlight,
        is_complaint,
        enriched_at
    from {{ source('ai', 'enriched_reviews') }}
),

restaurants as (
    select
        restaurant_id,
        restaurant_name,
        city,
        cuisine
    from {{ ref('dim_restaurants') }}
)

select
    r.review_id,
    r.order_id,
    r.customer_id,
    r.restaurant_id,
    res.restaurant_name,
    res.city,
    res.cuisine,
    r.rating,
    r.comment,
    r.review_date,
    r.sentiment,
    r.sentiment_score,
    r.primary_aspect,
    r.tags,
    r.key_issue_or_highlight,
    r.is_complaint,
    r.enriched_at
from enriched_reviews r
left join restaurants res on r.restaurant_id = res.restaurant_id
