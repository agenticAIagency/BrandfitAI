from typing import Annotated
from uuid import UUID

from fastapi import Header

from brandfit_core.config import get_settings


def workspace_id(x_workspace_id: Annotated[str | None, Header()] = None) -> UUID:
    return UUID(x_workspace_id) if x_workspace_id else get_settings().default_workspace_id
