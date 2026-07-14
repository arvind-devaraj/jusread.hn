import time
from concurrent.futures import ThreadPoolExecutor

URL = "https://hn.algolia.com/api/v1/search_by_date"
HITS_PER_PAGE = 1000  # max allowed per query
LOOKBACK_DAYS = 120
DAY_SECONDS = 24 * 60 * 60


def _fetch_day(session, day_start, day_end):
    params = {
        'tags': 'story',
        'numericFilters': f'created_at_i>{day_start},created_at_i<{day_end}',
        'page': 0,
        'hitsPerPage': HITS_PER_PAGE
    }
    response = session.get(URL, params=params)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None
    return response.json()


def get_past_month_hn_posts(session, days=LOOKBACK_DAYS, start_day=0):
    # The Algolia endpoint caps total retrievable results at ~1000 per query
    # regardless of nbHits, so a single 30-day query would silently truncate
    # to the newest ~1000 stories. Querying day-by-day keeps each request
    # under that cap. start_day lets callers fetch a specific slice (e.g.
    # days 7-29) instead of always starting from today.
    now = int(time.time())
    day_ranges = [
        (now - (i + 1) * DAY_SECONDS, now - i * DAY_SECONDS)
        for i in range(start_day, start_day + days)
    ]

    posts = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(
            lambda r: _fetch_day(session, r[0], r[1]),
            day_ranges
        )
        for data in results:
            if data:
                posts.extend(data.get('hits', []))

    return posts
