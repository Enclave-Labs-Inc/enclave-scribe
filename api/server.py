import uvicorn
from fastapi import FastAPI

from .routes.ocr import router as ocr_router
from .schemas import HealthResponse

app = FastAPI(
    title="EnclaveScribe",
    description="Document OCR API by Enclave Labs",
    version="0.1.0",
)

app.include_router(ocr_router)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


def main():
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
