"""Read the ARC-AGI-3 Kaggle forum through kagglesdk's discussions API.

  forum.py topics [query] [--sort hot|recent|top|...]   list topics (id, date, author, votes, comments, title)
  forum.py topic ID                                      full topic body + all comments -> topic-ID.md
"""
import sys, json, datetime
from pathlib import Path
from kagglesdk import KaggleClient
from kagglesdk.discussions.types.discussions_api_service import (
    ApiListTopicsRequest, ApiGetTopicRequest, ApiListCommentsRequest, ApiListForumsRequest)

HERE = Path(__file__).parent.parent / "forum"
FORUM = "arc-prize-2026-arc-agi-3"

def client():
    return KaggleClient().discussions.discussion_api_client

def topics(query="", sort=None, pages=5):
    c = client(); out = []; tok = ""
    for _ in range(pages):
        r = ApiListTopicsRequest(); r.forum_slug = FORUM; r.page_size = 100
        if query: r.search_query = query
        if sort:
            from kagglesdk.discussions.types.discussions_api_service import TopicListSortBy
            r.sort_by = getattr(TopicListSortBy, sort)
        if tok: r.page_token = tok
        resp = c.list_topics(r)
        out += resp.topics or []
        tok = getattr(resp, "next_page_token", "")
        if not tok: break
    return out

def walk(comments, depth=0):
    for cm in comments or []:
        yield depth, cm
        yield from walk(cm.replies, depth + 1)

def topic(tid):
    c = client()
    r = ApiGetTopicRequest(); r.id = int(tid)
    t = c.get_topic(r).topic
    lines = [f"# {t.title}", f"{t.author_name} · {t.post_date} · votes {t.votes} · {t.url}", "", t.content or "", ""]
    tok = ""
    while True:
        lr = ApiListCommentsRequest(); lr.topic_id = int(tid); lr.page_size = 100
        if tok: lr.page_token = tok
        resp = c.list_comments(lr)
        for d, cm in walk(resp.comments):
            lines += ["", f"{'>' * (d + 1)} **{cm.author_name}** · {cm.post_date} · votes {cm.votes}", "", cm.content or ""]
        tok = resp.next_page_token
        if not tok: break
    p = HERE / f"topic-{tid}.md"; p.write_text("\n".join(lines)); return p

if __name__ == "__main__":
    if sys.argv[1] == "topics":
        q = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
        sort = sys.argv[sys.argv.index("--sort") + 1] if "--sort" in sys.argv else None
        for t in topics(q, sort):
            print(t.id, str(t.post_date)[:10], f"{t.author_name[:22]:22s}", f"v{t.votes:<4d} c{t.comment_count:<4d}", t.title)
    else:
        print(topic(sys.argv[2]))


def search(query, pages=3, kind="DISCUSSIONS"):
    """Site search restricted to discussions; returns (id, date, author, votes, forum, title, url)."""
    from kagglesdk.search.types.search_api_service import ListEntitiesRequest, ListEntitiesFilters
    from kagglesdk.search.types import search_enums
    c = KaggleClient().search.search_api_client
    out, tok = [], ""
    for _ in range(pages):
        r = ListEntitiesRequest(); f = ListEntitiesFilters(); f.query = query
        dt = [d for d in search_enums.DocumentType if d.name.endswith(kind)]
        f.document_types = dt; r.filters = f; r.page_size = 50
        if tok: r.page_token = tok
        resp = c.list_entities(r)
        for d in resp.documents or []:
            dd = d.discussion_document
            out.append((d.id, str(d.create_time)[:10], (d.owner_user.user_name if d.owner_user else ""), d.votes,
                        getattr(dd, "forum_name", ""), d.title, getattr(dd, "new_comment_url", "")))
        tok = resp.next_page_token
        if not tok: break
    return out
