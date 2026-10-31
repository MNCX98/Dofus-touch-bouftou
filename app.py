"""Fenêtre principale (PySide6)."""
import os
import sys
from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QMainWindow, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from core import input as inp
from core.capture import WindowCapture
from core.config import load_config, save_config
from core.paths import captures_dir, config_path, logs_dir, monsters_dir
from core.vision import load_templates
from phases.base import Context
from phases.p1_recherche import run_loop, save_png, scan_once

VERSION = "0.1.0"


class Worker(QThread):
    """Exécute une tâche hors du thread de l'interface."""
    log = Signal(str)
    frame = Signal(object)

    def __init__(self, ctx: Context, mode: str):
        super().__init__()
        self.ctx, self.mode = ctx, mode
        self._stop = False
        ctx.log = self.log.emit

    def request_stop(self):
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop or inp.is_key_down(inp.VK_F12)

    def run(self):
        try:
            if self.mode == "loop":
                run_loop(self.ctx, self._should_stop, self.frame.emit)
            elif self.mode == "test":
                self.frame.emit(scan_once(self.ctx, do_click=False).frame)
            elif self.mode == "snapshot":
                img, _ = self.ctx.capture.grab()
                name = datetime.now().strftime("capture_%Y%m%d_%H%M%S.png")
                save_png(captures_dir() / name, img)
                self.log.emit(f"Capture enregistrée : captures/{name}  ({img.shape[1]}×{img.shape[0]} px)")
                self.frame.emit(img)
        except Exception as e:
            self.log.emit(f"⚠ {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"DofusAuto {VERSION} — Phase 1 : recherche des monstres")
        self.resize(1000, 700)
        self.cfg = load_config(config_path())
        self.ctx = Context(self.cfg, WindowCapture(self.cfg.window_title_contains))
        self.worker: Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.info = QLabel()
        root.addWidget(self.info)

        self.preview = QLabel("L'aperçu de ce que le programme voit apparaîtra ici.")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 340)
        self.preview.setStyleSheet("background:#222;color:#aaa;")
        root.addWidget(self.preview, 1)

        row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Démarrer")
        self.btn_stop = QPushButton("■ Arrêter (F12)")
        self.btn_test = QPushButton("Tester la détection (sans clic)")
        self.btn_snap = QPushButton("Enregistrer une capture")
        for b in (self.btn_start, self.btn_stop, self.btn_test, self.btn_snap):
            row.addWidget(b)
        root.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Seuil de confiance :"))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.50, 0.99)
        self.spin.setSingleStep(0.01)
        self.spin.setDecimals(2)
        self.spin.setValue(self.cfg.threshold)
        self.spin.valueChanged.connect(self._threshold_changed)
        row2.addWidget(self.spin)
        row2.addStretch(1)
        self.btn_reload = QPushButton("Recharger les images")
        self.btn_imgs = QPushButton("Dossier des images")
        self.btn_logs = QPushButton("Dossier des journaux")
        self.btn_caps = QPushButton("Dossier des captures")
        for b in (self.btn_reload, self.btn_imgs, self.btn_logs, self.btn_caps):
            row2.addWidget(b)
        root.addLayout(row2)

        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(500)
        self.logbox.setMaximumHeight(160)
        root.addWidget(self.logbox)

        self.btn_start.clicked.connect(lambda: self._launch("loop"))
        self.btn_test.clicked.connect(lambda: self._launch("test"))
        self.btn_snap.clicked.connect(lambda: self._launch("snapshot"))
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reload.clicked.connect(self._reload_templates)
        self.btn_imgs.clicked.connect(lambda: self._open(monsters_dir()))
        self.btn_logs.clicked.connect(lambda: self._open(logs_dir()))
        self.btn_caps.clicked.connect(lambda: self._open(captures_dir()))

        self._reload_templates()
        self._set_running(False)
        QTimer.singleShot(0, lambda: self.log("Prêt. Place BlueStacks bien visible (sans le recouvrir), "
                                              "puis clique sur « Tester la détection »."))

    # --- utilitaires -------------------------------------------------
    def log(self, msg: str):
        self.logbox.appendPlainText(f"[{datetime.now():%H:%M:%S}] {msg}")

    def _open(self, path):
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))

    def _threshold_changed(self, v: float):
        self.cfg.threshold = round(v, 2)

    def _reload_templates(self):
        self.ctx.templates = load_templates(monsters_dir())
        monsters = {}
        for t in self.ctx.templates:
            monsters.setdefault(t.monster, 0)
            monsters[t.monster] += 1
        txt = ", ".join(f"{m} ({n} variante{'s' if n > 1 else ''})" for m, n in monsters.items())
        self.info.setText(f"Monstres chargés : {txt or 'aucun — ajoute des images dans le dossier des images'}")
        self.log(f"{len(self.ctx.templates)} image(s) de référence chargée(s).")

    def _set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_test.setEnabled(not running)
        self.btn_snap.setEnabled(not running)
        self.btn_reload.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    def _launch(self, mode: str):
        if self.worker and self.worker.isRunning():
            return
        self.ctx.capture.title_contains = self.cfg.window_title_contains
        self.worker = Worker(self.ctx, mode)
        self.worker.log.connect(self.log)
        self.worker.frame.connect(self._show_frame)
        self.worker.finished.connect(lambda: self._set_running(False))
        self._set_running(True)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.request_stop()

    def _show_frame(self, img: np.ndarray):
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        pix = QPixmap.fromImage(qimg).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def closeEvent(self, event):
        self._stop()
        if self.worker:
            self.worker.wait(3000)
        try:
            save_config(self.cfg, config_path())
        except OSError:
            pass
        event.accept()


def run():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
