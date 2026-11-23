"""Original metadata-only nearest-completed Stage3 rule, with cold fallback."""
from datetime import datetime,timedelta,timezone


def select_support(query,candidates):
    if query['layout']!='1x5x1':return None,'query_not_CMCU'
    prior=[r for r in candidates if r['stage']==3 and r['subject']==query['subject'] and r['hand']==query['hand'] and r['start']<query['start'] and r['start']+r['duration']<=query['start']]
    if not prior:return None,'no_prior_completed_same_hand_stage3'
    support=max(prior,key=lambda r:(r['start'],r['key']))
    zone=timezone(timedelta(hours=8))
    if query['start']-support['start']>7200 or datetime.fromtimestamp(query['start'],zone).date()!=datetime.fromtimestamp(support['start'],zone).date():return None,'outside_2h_or_date'
    if support['layout']!='1x5x1' or support['n']<120000:return None,'nearest_support_ineligible_no_fallback'
    if query.get('content_hash') and query['content_hash']==support.get('content_hash'):return None,'same_content'
    return support,'supported'
