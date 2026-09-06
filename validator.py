import os
import hashlib
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Validator:

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    MaxImageSize = 1024 * 1024 * 20 #20MB

    @staticmethod
    def load_image_safely(file_path: str):
        try:
            img_data = np.fromfile(file_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                return None
            return cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning("Failed to read image file '%s': %s", file_path, e)
            return None

    @staticmethod
    def is_valid_image(file_path: str) -> tuple[bool, str]:
        if not file_path or not os.path.exists(file_path):
            return False, "File path is invalid or does not exist."

        if not os.path.isfile(file_path):
            return False, "The specified path is not a valid file."

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in Validator.image_extensions:
            return False, f"Invalid file extension ({ext}). Please select an image (JPG, PNG, BMP, TIFF, WEBP)."

        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            return False, "Unable to read file information."

        if file_size == 0:
            return False, "The file is empty and contains no data."

        if file_size > Validator.MaxImageSize:
            size_mb = file_size / (1024 * 1024)
            max_mb = Validator.MaxImageSize / (1024 * 1024)
            return False, f"File size ({size_mb:.1f}MB) exceeds the maximum limit of {max_mb:.0f}MB."

        img = Validator.load_image_safely(file_path)
        if img is None or img.size == 0:
            return False, "Unable to decode the image. The file may be corrupted or not a valid image."

        return True, "Valid image."

    @staticmethod
    def calculate_hash(file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def is_same_image(file_path1: str, file_path2: str) -> tuple[bool, str]:
        if not os.path.exists(file_path1) or not os.path.exists(file_path2):
            return False, "One or both file paths do not exist."

        if os.path.abspath(file_path1) == os.path.abspath(file_path2):
            return True, "Images are identical (same file path)."

        try:
            hash1 = Validator.calculate_hash(file_path1)
            hash2 = Validator.calculate_hash(file_path2)
        except OSError:
            return False, "Unable to read one of the files while comparing content."

        if hash1 == hash2:
            return True, "Images are the same content."
        return False, "Images are different."

