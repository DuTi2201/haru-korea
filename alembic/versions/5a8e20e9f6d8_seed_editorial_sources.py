"""seed verified editorial_source rows (opinion/editorial RSS)

Owner asked for KBS/Chosun/Naver specifically, but research this session
found:
  - Naver News RSS was fully discontinued (community reports it stopped
    working entirely in March; there is no current replacement), so it
    is not seeded at all.
  - KBS (broadcaster) does not appear to publish a public RSS feed the
    way newspapers do — none could be found or verified, so it is not
    seeded either. The admin can add one later via POST
    /editorial-sources if a working KBS feed turns up.
  - Chosun Ilbo *is* included: it runs on the Arc Publishing platform,
    whose outbound-feed URL convention is well documented
    (https://www.chosun.com/arc/outboundfeeds/rss/category/opinion/
    ?outputType=xml). This session's web-fetch tool could not reach
    chosun.com directly to confirm the feed is currently live (the
    domain is blocked for that tool), so this one is unverified — but
    discover_editorial_candidates already isolates a single bad
    source's errors (see app/services/ingestion.py) without affecting
    the others, so seeding it "active" is a safe bet worth testing.

In their place, two newspapers' dedicated 오피니언/사설 (opinion/
editorial) RSS feeds were fetched directly this session and confirmed
live with current dated entries:
  - Kyunghyang Shinmun (경향신문) — opinion feed, entries include
    [사설]-tagged editorials, freshest article same-day.
  - Segye Ilbo (세계일보) — opinion feed, valid RSS 2.0, dated entries.

All three rows are `active`; discover_editorial_candidates (Celery
beat, every 6h) will keyword-filter these for the topics the admin
already configured (aging population/AI/climate/education) and stage
matching candidates for review — never auto-publish, per the SDD.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5a8e20e9f6d8'
down_revision = 'f3c8a1d9e4b2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    editorial_source = sa.table(
        'editorial_source',
        sa.column('name', sa.String),
        sa.column('base_url', sa.String),
        sa.column('rss_url', sa.String),
        sa.column('license_note', sa.Text),
        sa.column('active', sa.Boolean),
        schema='editorial',
    )
    op.bulk_insert(
        editorial_source,
        [
            {
                'name': 'Kyunghyang Shinmun (경향신문) - Opinion',
                'base_url': 'https://www.khan.co.kr',
                'rss_url': 'https://www.khan.co.kr/rss/rssdata/opinion_news.xml',
                'license_note': (
                    'Official opinion/editorial RSS, verified live this session '
                    '(same-day [사설] editorial entries present at seed time). '
                    'Family-internal use only, per owner instruction.'
                ),
                'active': True,
            },
            {
                'name': 'Segye Ilbo (세계일보) - Opinion',
                'base_url': 'https://www.segye.com',
                'rss_url': 'http://www.segye.com/Articles/RSSList/segye_opinion.xml',
                'license_note': (
                    'Official opinion/editorial RSS, verified live this session '
                    '(valid RSS 2.0, dated entries). Family-internal use only, '
                    'per owner instruction.'
                ),
                'active': True,
            },
            {
                'name': 'Chosun Ilbo (조선일보) - Opinion',
                'base_url': 'https://www.chosun.com',
                'rss_url': 'https://www.chosun.com/arc/outboundfeeds/rss/category/opinion/?outputType=xml',
                'license_note': (
                    'UNVERIFIED this session: chosun.com is blocked for this '
                    "session's web-fetch tool, so this URL (Arc Publishing's "
                    'standard outbound-feed convention, which chosun.com runs '
                    'on) could not be fetched directly to confirm it is live. '
                    'A single bad source only shows up in '
                    'discover_editorial_candidates\' per-source errors and '
                    "does not affect the other sources, so it's seeded active "
                    'to test — deactivate via PATCH /editorial-sources/{id} '
                    'if it turns out to 404. Family-internal use only, per '
                    'owner instruction.'
                ),
                'active': True,
            },
        ],
    )


def downgrade() -> None:
    op.execute(
        "DELETE FROM editorial.editorial_source WHERE name IN ("
        "'Kyunghyang Shinmun (경향신문) - Opinion', "
        "'Segye Ilbo (세계일보) - Opinion', "
        "'Chosun Ilbo (조선일보) - Opinion'"
        ")"
    )
