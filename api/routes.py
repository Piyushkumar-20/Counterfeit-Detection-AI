import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.decision.review_queue import enqueue_for_review
from src.hardware.bridge import decode_image_bytes, process_hardware_capture
from src.main import process_image
from src.ops.metrics import collect_operational_metrics


router = APIRouter()


@router.get("/health")
def health_check():
	return {"status": "ok"}


@router.get("/ops/metrics")
def ops_metrics():
	return collect_operational_metrics()


@router.post("/verify")
async def verify_package(
	image: UploadFile = File(...),
	uv_image: Optional[UploadFile] = File(default=None),
	debug: bool = Form(default=False),
):
	if image.content_type and not image.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="image must be a valid image file")

	normal_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
	uv_temp = None

	try:
		normal_temp.write(await image.read())
		normal_temp.flush()
		normal_path = normal_temp.name

		uv_path = None
		if uv_image is not None:
			uv_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
			uv_temp.write(await uv_image.read())
			uv_temp.flush()
			uv_path = uv_temp.name

		result = process_image(normal_image_path=normal_path, uv_image_path=uv_path, debug=debug)

		decision = result["decision"]
		if decision.get("confidence", 0.0) < 0.55:
			enqueue_for_review(result, reason="low-confidence-api-verification")

		return {
			"drug_name": decision.get("drug_name"),
			"dosage": decision.get("dosage"),
			"qr_status": result["qr"],
			"uv_score": result["uv"].get("similarity", 0.0),
			"image_dataset_match": result.get("image_dataset_match"),
			"regulatory_sources": result.get("regulatory_sources"),
			"final_decision": decision.get("final_decision"),
			"confidence": decision.get("confidence"),
			"probability_authentic": decision.get("probability_authentic"),
			"reasoning": decision.get("reasoning"),
			"feature_breakdown": decision.get("feature_breakdown"),
			"regulatory_assessment": decision.get("regulatory_assessment"),
		}
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc
	finally:
		normal_temp.close()
		if uv_temp is not None:
			uv_temp.close()

		if os.path.exists(normal_temp.name):
			os.remove(normal_temp.name)
		if uv_temp is not None and os.path.exists(uv_temp.name):
			os.remove(uv_temp.name)


@router.post("/verify/hardware")
async def verify_hardware_capture(
	normal_image: UploadFile = File(...),
	uv_image: Optional[UploadFile] = File(default=None),
	session_id: Optional[str] = Form(default=None),
	camera_id: Optional[str] = Form(default=None),
	debug: bool = Form(default=False),
):
	if normal_image.content_type and not normal_image.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="normal_image must be a valid image file")

	if uv_image is not None and uv_image.content_type and not uv_image.content_type.startswith("image/"):
		raise HTTPException(status_code=400, detail="uv_image must be a valid image file")

	try:
		normal_frame = decode_image_bytes(await normal_image.read())
		uv_frame = decode_image_bytes(await uv_image.read()) if uv_image is not None else None
		if normal_frame is None:
			raise HTTPException(status_code=400, detail="normal_image could not be decoded")

		metadata = {
			"session_id": session_id,
			"camera_id": camera_id,
			"ingest_mode": "hardware",
		}
		result = process_hardware_capture(normal_frame=normal_frame, uv_frame=uv_frame, metadata=metadata, debug=debug)
		decision = result["decision"]
		if decision.get("confidence", 0.0) < 0.55:
			enqueue_for_review(result, reason="low-confidence-hardware-verification")

		return {
			"hardware_metadata": result.get("hardware_metadata", {}),
			"decision": decision,
			"qr": result.get("qr", {}),
			"uv": result.get("uv", {}),
			"image_dataset_match": result.get("image_dataset_match"),
			"regulatory_sources": result.get("regulatory_sources"),
		}
	except HTTPException:
		raise
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Hardware verification failed: {exc}") from exc
