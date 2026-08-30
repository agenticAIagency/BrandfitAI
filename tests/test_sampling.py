from datetime import UTC, datetime, timedelta
from uuid import uuid4

from brandfit_core.agents.planner import _sample_posts
from brandfit_core.db.models import Post
from brandfit_core.domain.enums import MediaType


def test_adaptive_sample_keeps_latest_and_pinned_and_caps_at_50() -> None:
    posts = [
        Post(
            id=uuid4(),
            workspace_id=uuid4(),
            creator_id=uuid4(),
            ingestion_batch_id=uuid4(),
            platform="fixture",
            platform_post_id=str(index),
            media_type=MediaType.IMAGE if index % 2 else MediaType.REEL,
            published_at=datetime.now(UTC) - timedelta(days=index),
            is_pinned=index == 55,
        )
        for index in range(70)
    ]
    selected = _sample_posts(posts, {})
    selected_ids = {item.id for item in selected}
    assert len(selected) == 50
    assert {item.id for item in posts[:20]} <= selected_ids
    assert posts[55].id in selected_ids
