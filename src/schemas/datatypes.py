from typing import List, Tuple, Union

from src.schemas.dataclasses import PipelineEntry, SpacyEntry

BertPipelineData = List[PipelineEntry]
PredictedData = List[Union[SpacyEntry, PipelineEntry]]
PredictedFormat = List[Tuple[str, str]]
RawSpacyData = List[SpacyEntry]
