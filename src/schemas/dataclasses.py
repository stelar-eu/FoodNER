from typing import TypedDict


class PipelineEntry(TypedDict):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


class SpacyEntry(TypedDict):
    start: int
    end: int
    label: str
    text: str
