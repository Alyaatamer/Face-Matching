import os
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QPainter, QPen, QColor, QImage
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QFrame, QProgressBar
)

from validator import Validator
from face_detector import FaceDetector
from comparator import Comparator

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
PREVIEW_SIZE = 260


class CompareWorker(QThread):
    status_changed = Signal(str)
    face_detected = Signal(int, dict)
    error_occurred = Signal(str, str)
    finished = Signal(dict)

    def __init__(self, path1: str, path2: str):
        super().__init__()
        self.path1 = path1
        self.path2 = path2

    def run(self):
        self.status_changed.emit("Analyzing Face 1 for face detection...")
        count1, msg1, area1, crop1 = FaceDetector.detect_and_count(self.path1)
        if count1 != 1:
            self.error_occurred.emit("Face Detection Error - Face 1", msg1)
            return
        if area1:
            self.face_detected.emit(1, area1)

        self.status_changed.emit("Analyzing Face 2 for face detection...")
        count2, msg2, area2, crop2 = FaceDetector.detect_and_count(self.path2)
        if count2 != 1:
            self.error_occurred.emit("Face Detection Error - Face 2", msg2)
            return
        if area2:
            self.face_detected.emit(2, area2)

        self.status_changed.emit("Verifying facial identity with ArcFace...")
        result = Comparator.compare_faces(self.path1, self.path2, face1_crop=crop1, face2_crop=crop2)
        self.finished.emit(result)


class ImageSlot(QFrame):
    image_selected = Signal(str)

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.file_path = None
        self.setAcceptDrops(True)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            ImageSlot {
                background-color: #FAFAFA;
                border: 2px dashed #CBD5E1;
                border-radius: 10px;
            }
            ImageSlot:hover {
                border-color: #3B82F6;
                background-color: #F8FAFC;
            }
        """)

        self.title_label = QLabel(f"<b>{title}</b>")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 14px; color: #1E293B; margin-top: 4px;")

        self.preview = QLabel("Drag & Drop Image Here\nor click Select")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(PREVIEW_SIZE, PREVIEW_SIZE)
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("""
            QLabel {
                background-color: #F1F5F9;
                color: #64748B;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 500;
            }
        """)

        self.info_label = QLabel("No image selected")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #94A3B8; font-size: 11px;")

        self.select_button = QPushButton(f"Select {title}")
        self.select_button.setCursor(Qt.PointingHandCursor)
        self.select_button.setFixedHeight(34)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #FFFFFF;
                color: #0F172A;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
                border-color: #94A3B8;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.preview, alignment=Qt.AlignCenter)
        layout.addWidget(self.info_label)
        layout.addWidget(self.select_button)
        self.setLayout(layout)

    def set_image(self, file_path: str):
        self.file_path = file_path
        img_bgr = Validator.load_image_safely(file_path)
        if img_bgr is not None:
            h, w, ch = img_bgr.shape
            bytes_per_line = ch * w
            qimg = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(
                PREVIEW_SIZE - 4, PREVIEW_SIZE - 4,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview.setPixmap(scaled)
            filename = os.path.basename(file_path)
            self.info_label.setText(f"{filename} ({w}x{h})")
            self.info_label.setStyleSheet("color: #334155; font-size: 11px; font-weight: bold;")
        self.image_selected.emit(file_path)

    def clear(self):
        self.file_path = None
        self.preview.clear()
        self.preview.setText("Drag & Drop Image Here\nor click Select")
        self.info_label.setText("No image selected")
        self.info_label.setStyleSheet("color: #94A3B8; font-size: 11px;")

    def highlight_face(self, facial_area: dict):
        if not self.file_path or not facial_area:
            return

        img_bgr = Validator.load_image_safely(self.file_path)
        if img_bgr is None:
            return

        h, w, ch = img_bgr.shape
        bytes_per_line = ch * w
        qimg = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen_width = max(4, int(min(pixmap.width(), pixmap.height()) * 0.008))
        pen = QPen(QColor("#22C55E"), pen_width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        fx = int(facial_area.get('x', 0))
        fy = int(facial_area.get('y', 0))
        fw = int(facial_area.get('w', 0))
        fh = int(facial_area.get('h', 0))

        if fw <= 0 and 'x2' in facial_area:
            fx = int(facial_area['x1'])
            fy = int(facial_area['y1'])
            fw = int(facial_area['x2']) - fx
            fh = int(facial_area['y2']) - fy

        margin_x = int(fw * 0.08)
        margin_y = int(fh * 0.08)
        rx = max(0, fx - margin_x)
        ry = max(0, fy - margin_y)
        rw = min(pixmap.width() - rx, fw + 2 * margin_x)
        rh = min(pixmap.height() - ry, fh + 2 * margin_y)

        corner_radius = int(min(rw, rh) * 0.3)
        painter.drawRoundedRect(rx, ry, rw, rh, corner_radius, corner_radius)
        painter.end()

        scaled = pixmap.scaled(
            PREVIEW_SIZE - 4, PREVIEW_SIZE - 4,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(scaled)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                ext = os.path.splitext(urls[0].toLocalFile())[1].lower()
                if ext in Validator.image_extensions:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            local_path = urls[0].toLocalFile()
            is_valid, msg = Validator.is_valid_image(local_path)
            if is_valid:
                self.set_image(local_path)
                event.acceptProposedAction()
            else:
                QMessageBox.warning(self.window(), "Invalid Image", msg)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Matching Tool")
        self.worker = None

        self.setStyleSheet("""
            QWidget {
                background-color: #F8FAFC;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }
        """)

        self.slot1 = ImageSlot("Face 1")
        self.slot2 = ImageSlot("Face 2")
        self.slot1.select_button.clicked.connect(lambda: self.select_image_dialog(self.slot1))
        self.slot2.select_button.clicked.connect(lambda: self.select_image_dialog(self.slot2))
        self.slot1.image_selected.connect(self.clear_result)
        self.slot2.image_selected.connect(self.clear_result)

        images_layout = QHBoxLayout()

        images_layout.setSpacing(20)
        images_layout.addWidget(self.slot1)
        images_layout.addWidget(self.slot2)

        self.compare_button = QPushButton("Compare")
        self.compare_button.setFixedHeight(44)
        self.compare_button.setCursor(Qt.PointingHandCursor)
        self.compare_button.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:disabled {
                background-color: #94A3B8;
            }
        """)
        self.compare_button.clicked.connect(self.on_compare_clicked)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedHeight(44)
        self.reset_button.setFixedWidth(90)
        self.reset_button.setCursor(Qt.PointingHandCursor)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #E2E8F0;
                color: #334155;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #CBD5E1;
            }
        """)
        self.reset_button.clicked.connect(self.on_reset_clicked)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        buttons_layout.addWidget(self.compare_button, stretch=1)
        buttons_layout.addWidget(self.reset_button)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #475569; font-size: 12px; font-weight: 500;")

        self.result_card = QFrame()
        self.result_card.setFrameShape(QFrame.StyledPanel)
        self.result_card.setVisible(False)
        self.result_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
            }
        """)

        self.decision_label = QLabel("")
        self.decision_label.setAlignment(Qt.AlignCenter)
        self.decision_label.setStyleSheet("font-size: 22px; font-weight: 800;")

        self.metric_label = QLabel("")
        self.metric_label.setAlignment(Qt.AlignCenter)
        self.metric_label.setStyleSheet("font-size: 12px; color: #475569;")

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(16, 12, 16, 12)
        card_layout.addWidget(self.decision_label)
        card_layout.addWidget(self.metric_label)
        self.result_card.setLayout(card_layout)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)
        main_layout.addLayout(images_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.result_card)
        self.setLayout(main_layout)
        self.setMinimumSize(660, 520)

    def select_image_dialog(self, slot: ImageSlot):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {slot.title}", "", IMAGE_FILTER)
        if not file_path:
            return

        is_valid, message = Validator.is_valid_image(file_path)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Image", message)
            return

        slot.set_image(file_path)

    def clear_result(self):
        self.result_card.setVisible(False)
        self.status_label.setText("")

    def on_reset_clicked(self):
        self.slot1.clear()
        self.slot2.clear()
        self.clear_result()

    def on_compare_clicked(self):
        path1 = self.slot1.file_path
        path2 = self.slot2.file_path

        if not path1 or not path2:
            QMessageBox.warning(self, "Missing Images", "Please select both Face 1 and Face 2 before comparing.")
            return

        same, _ = Validator.is_same_image(path1, path2)
        if same:
            reply = QMessageBox.question(
                self, "Same Image",
                "Both selected images appear to be identical. Continue comparison?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        self.compare_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.compare_button.setText("Processing...")
        self.status_label.setText("Initializing comparison pipeline...")
        self.result_card.setVisible(False)

        self.worker = CompareWorker(path1, path2)
        self.worker.status_changed.connect(self.on_worker_status)
        self.worker.face_detected.connect(self.on_face_detected)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_worker_status(self, text: str):
        self.status_label.setText(text)

    def on_face_detected(self, slot_num: int, area: dict):
        if slot_num == 1:
            self.slot1.highlight_face(area)
        elif slot_num == 2:
            self.slot2.highlight_face(area)

    def on_worker_error(self, title: str, message: str):
        self.compare_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.compare_button.setText("Compare")
        self.status_label.setText("")
        QMessageBox.warning(self, title, message)

    def on_worker_finished(self, result: dict):
        self.compare_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.compare_button.setText("Compare")
        self.status_label.setText("")

        is_match = result.get("is_match", False)
        distance = result.get("distance", -1.0)
        threshold = result.get("threshold", 0.68)
        similarity_pct = result.get("similarity_pct", 0.0)
        from_cache = result.get("from_cache", False)

        if distance < 0:
            QMessageBox.critical(self, "Comparison Failed", result.get("message", "Unknown error."))
            return

        self.result_card.setVisible(True)
        cache_note = " • (from cache)" if from_cache else ""

        if is_match:
            self.result_card.setStyleSheet("""
                QFrame {
                    background-color: #F0FDF4;
                    border: 2px solid #22C55E;
                    border-radius: 10px;
                }
            """)
            self.decision_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #15803D;")
            self.decision_label.setText("✅ Match")
            self.metric_label.setText(
                f"Similarity Score: <b>{similarity_pct:.1f}%</b>  |  "
                f"Distance: {distance:.4f} (Threshold: {threshold:.4f}){cache_note}"
            )
        else:
            self.result_card.setStyleSheet("""
                QFrame {
                    background-color: #FEF2F2;
                    border: 2px solid #EF4444;
                    border-radius: 10px;
                }
            """)
            self.decision_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #B91C1C;")
            self.decision_label.setText("❌ No Match")
            self.metric_label.setText(
                f"Similarity Score: <b>{similarity_pct:.1f}%</b>  |  "
                f"Distance: {distance:.4f} (Threshold: {threshold:.4f}){cache_note}"
            )
