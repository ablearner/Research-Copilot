import logging
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, Request, UploadFile, status

from apps.api.exception_handlers import build_error_detail
from core.config import get_settings
from domain.schemas.api import UploadDocumentResponse

logger = logging.getLogger(__name__)

_UPLOAD_CHUNK_SIZE = 1024 * 1024
_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
}


def upload_preview_enabled(settings: object) -> bool:
    return str(getattr(settings, "app_env", "local")).strip().lower() == "local"


async def read_upload_chunk(file: UploadFile, size: int) -> bytes:
    raw_file = getattr(file, "file", None)
    if isinstance(raw_file, BytesIO):
        return raw_file.read(size)
    return await file.read(size)


async def handle_upload_document_request(
    request: Request,
    file: UploadFile,
) -> UploadDocumentResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    document_id = f"doc_{uuid4().hex}"
    suffix = Path(file.filename or "document").suffix
    settings = get_settings()
    upload_dir = settings.resolve_path(settings.upload_dir)
    target_path = upload_dir / f"{document_id}{suffix}"
    max_upload_bytes = max(1, int(getattr(settings, "upload_max_bytes", 25 * 1024 * 1024)))
    if file.size is not None and file.size > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds max size of {max_upload_bytes} bytes",
        )

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        bytes_written = 0
        with target_path.open("wb") as output:
            while True:
                chunk = await read_upload_chunk(file, _UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Uploaded file exceeds max size of {max_upload_bytes} bytes",
                    )
                output.write(chunk)
        logger.info(
            "Uploaded document",
            extra={
                "document_id": document_id,
                "uploaded_filename": file.filename,
                "content_type": file.content_type,
            },
        )
        metadata: dict[str, str] = {}
        if upload_preview_enabled(settings):
            metadata["preview_url"] = f"/uploads/{target_path.name}"
        return UploadDocumentResponse(
            document_id=document_id,
            filename=file.filename or target_path.name,
            status="uploaded",
            storage_uri=str(target_path.resolve()),
            metadata=metadata,
        )
    except HTTPException:
        target_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target_path.unlink(missing_ok=True)
        logger.exception(
            "Failed to persist uploaded document",
            extra={"uploaded_filename": file.filename},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=build_error_detail(request, fallback="Failed to store uploaded document", exc=exc),
        ) from exc
    finally:
        await file.close()
