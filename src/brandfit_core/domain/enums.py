from enum import StrEnum


class BatchStatus(StrEnum):
    RECEIVED = "received"
    VALIDATING = "validating"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class MediaType(StrEnum):
    IMAGE = "image"
    CAROUSEL = "carousel"
    REEL = "reel"
    VIDEO = "video"


class ArtifactType(StrEnum):
    KEYFRAME = "keyframe"
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    OCR = "ocr"


class PersonaStatus(StrEnum):
    INSUFFICIENT = "insufficient"
    PRELIMINARY = "preliminary"
    ESTABLISHED = "established"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class WorkflowKind(StrEnum):
    INGESTION = "ingestion"
    POST_ANALYSIS = "post_analysis"
    CREATOR_PERSONA = "creator_persona"
    EXPORT = "export"
