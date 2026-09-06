import logging
from deepface import DeepFace
from validator import Validator

logger = logging.getLogger(__name__)

DETECTOR_BACKEND = "retinaface"


class FaceDetector:

    @staticmethod
    def detect_and_count(file_path: str) -> tuple[int, str, dict | None, any]:
        try:
            img_bgr = Validator.load_image_safely(file_path)
            if img_bgr is None:
                return 0, "Failed to decode the image file.", None, None

            img_rgb = img_bgr[:, :, ::-1]

            faces = DeepFace.extract_faces(
                img_path=img_rgb,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                normalize_face=False,
            )

            face_count = len(faces)
            if face_count > 1:
                return face_count, f"Multiple faces detected ({face_count} faces). Please use an image with only one person.", None, None

            primary_area = faces[0].get("facial_area") if faces else None
            primary_crop = faces[0].get("face") if faces else None
            return 1, "Single face detected successfully.", primary_area, primary_crop

        except ValueError:
            return 0, "No face detected in the image. Please choose a clear photo of a person.", None, None

        except Exception as e:
            logger.error("Error in detect_and_count for '%s': %s", file_path, e, exc_info=True)
            return 0, f"An unexpected error occurred during face detection: {str(e)}", None, None

    @staticmethod
    def count_faces(file_path: str) -> tuple[int, str]:
        count, msg, _, _ = FaceDetector.detect_and_count(file_path)
        return count, msg

