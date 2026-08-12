from typing import Literal

from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image bytes")
    mime_type: str = Field("image/png", description="MIME type of the image")
    prompt: str = Field("document parsing.", description="Instruction prompt sent to the model")
    clean_output: bool = Field(True, description="Strip detection markers from output")


class OCRResponse(BaseModel):
    text: str
    tokens: int
    status: Literal["ok", "error"] = "ok"


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
