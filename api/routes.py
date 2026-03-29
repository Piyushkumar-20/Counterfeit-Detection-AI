import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.main import process_image


router = APIRouter()


@router.get("/health")
def health_check():
	return {"status": "ok"}


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
		return {
			"drug_name": decision.get("drug_name"),
			"dosage": decision.get("dosage"),
			"qr_status": result["qr"],
			"uv_score": result["uv"].get("similarity", 0.0),
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
