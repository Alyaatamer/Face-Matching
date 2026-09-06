import logging
from deepface import DeepFace
from face_detector import DETECTOR_BACKEND
from validator import Validator

logger = logging.getLogger(__name__)

MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "cosine"

_COMPARISON_CACHE: dict[tuple[str, str], dict] = {}


class Comparator:

    @staticmethod
    def calculate_similarity_percent(distance: float, threshold: float) -> float:
        if distance < 0:
            return 0.0
        if threshold <= 0:
            return max(0.0, min(100.0, (1.0 - distance) * 100))

        if distance <= threshold:
            ratio = distance / threshold
            pct = 100.0 - (ratio * 40.0)
        else:
            over = (distance - threshold) / threshold
            pct = max(0.0, 60.0 - (over * 50.0))

        return round(float(pct), 1)

    @staticmethod
    def compare_faces(file_path1: str, file_path2: str, face1_crop=None, face2_crop=None) -> dict:
        try:
            h1 = Validator.calculate_hash(file_path1)
            h2 = Validator.calculate_hash(file_path2)
            cache_key = (min(h1, h2), max(h1, h2))
            if cache_key in _COMPARISON_CACHE:
                cached = dict(_COMPARISON_CACHE[cache_key])
                cached["from_cache"] = True
                cached["message"] = "Result retrieved from in-memory cache."
                logger.info("Cache hit for pair (%s, %s)", h1[:8], h2[:8])
                return cached
        except Exception as e:
            logger.warning("Could not calculate cache key: %s", e)
            cache_key = None

        try:
            if face1_crop is not None and face2_crop is not None:
                input1 = face1_crop
                input2 = face2_crop
                active_backend = "skip"
                enforce_det = False
            else:
                img1_bgr = Validator.load_image_safely(file_path1)
                img2_bgr = Validator.load_image_safely(file_path2)

                if img1_bgr is None or img2_bgr is None:
                    return {
                        "is_match": False,
                        "similarity_pct": 0.0,
                        "distance": -1.0,
                        "threshold": 0.0,
                        "message": "Unable to decode one of the image files.",
                        "from_cache": False,
                    }

                input1 = img1_bgr[:, :, ::-1]
                input2 = img2_bgr[:, :, ::-1]
                active_backend = DETECTOR_BACKEND
                enforce_det = True

            result = DeepFace.verify(
                img1_path=input1,
                img2_path=input2,
                model_name=MODEL_NAME,
                detector_backend=active_backend,
                distance_metric=DISTANCE_METRIC,
                enforce_detection=enforce_det
            )

            is_match = bool(result.get("verified", False))
            distance = float(result.get("distance", 1.0))
            threshold = float(result.get("threshold", 0.68))
            similarity_pct = Comparator.calculate_similarity_percent(distance, threshold)

            output = {
                "is_match": is_match,
                "similarity_pct": similarity_pct,
                "distance": round(distance, 4),
                "threshold": round(threshold, 4),
                "message": "Comparison completed successfully.",
                "from_cache": False,
            }

            if cache_key:
                _COMPARISON_CACHE[cache_key] = dict(output)

            return output

        except ValueError:
            return {
                "is_match": False,
                "similarity_pct": 0.0,
                "distance": -1.0,
                "threshold": 0.0,
                "message": "No face detected in one of the images during comparison.",
                "from_cache": False,
            }

        except Exception as e:
            logger.error("Unexpected error in compare_faces: %s", e, exc_info=True)
            return {
                "is_match": False,
                "similarity_pct": 0.0,
                "distance": -1.0,
                "threshold": 0.0,
                "message": f"An unexpected error occurred during comparison: {str(e)}",
                "from_cache": False,
            }

