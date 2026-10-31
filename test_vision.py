"""Tests de la reconnaissance sur des scènes synthétiques (fond différent à chaque fois).

Ils vérifient que le code fonctionne ; la fiabilité réelle se mesure sur de vraies
captures de BlueStacks.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.vision import load_templates, scan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "assets" / "monsters"
THRESHOLD = 0.80


def make_background(rng, w=1280, h=720):
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (rng.integers(30, 70), rng.integers(110, 170), rng.integers(60, 110))   # herbe (BGR)
    base += rng.normal(0, 14, (h, w, 1))
    for _ in range(40):                                      # taches de terre / décor
        x, y = rng.integers(0, w - 120), rng.integers(0, h - 90)
        cv2.rectangle(base, (x, y), (x + rng.integers(30, 120), y + rng.integers(20, 90)),
                      tuple(int(c) for c in rng.integers(40, 200, 3)), -1)
    base = cv2.GaussianBlur(base, (0, 0), 2)
    return np.clip(base, 0, 255).astype(np.uint8)


def paste(scene, tpl, x, y):
    h, w = tpl.image.shape[:2]
    m = tpl.mask[:, :, 0] > 0
    roi = scene[y:y + h, x:x + w]
    roi[m] = tpl.image[m]


def main():
    templates = load_templates(ROOT)
    assert len(templates) >= 4, f"Images de référence manquantes ({len(templates)} trouvées)"
    rng = np.random.default_rng(42)
    failures = 0

    # 1) chaque variante placée seule sur un décor différent
    for t in templates:
        scene = make_background(rng)
        x, y = int(rng.integers(50, 1000)), int(rng.integers(50, 560))
        paste(scene, t, x, y)
        scene = np.clip(scene.astype(np.float32) + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
        res = scan(scene, templates, THRESHOLD)
        ok = bool(res.detections) and abs(res.detections[0].x - x) <= 2 and abs(res.detections[0].y - y) <= 2
        ok = ok and res.detections[0].variant == t.variant
        print(f"[{'OK' if ok else 'ÉCHEC'}] {t.variant:24s} trouvé" +
              (f" score {res.detections[0].score:.2f}" if res.detections else " : rien"))
        failures += not ok

    # 2) scène sans monstre : aucune détection
    nothing = 0
    for _ in range(5):
        res = scan(make_background(rng), templates, THRESHOLD)
        nothing += len(res.detections)
    ok = nothing == 0
    print(f"[{'OK' if ok else 'ÉCHEC'}] 5 scènes vides : {nothing} fausse(s) détection(s)")
    failures += not ok

    # 3) plusieurs monstres : tous détectés, la liste est triée par score
    scene = make_background(rng)
    spots = [(100, 100), (600, 200), (900, 450)]
    for t, (x, y) in zip(templates[:3], spots):
        paste(scene, t, x, y)
    res = scan(scene, templates, THRESHOLD)
    ok = len(res.detections) == 3 and res.detections == sorted(res.detections, key=lambda d: -d.score)
    print(f"[{'OK' if ok else 'ÉCHEC'}] 3 monstres sur la carte : {len(res.detections)} détecté(s)")
    failures += not ok

    # 4) vitesse
    t0 = time.time()
    for _ in range(5):
        scan(scene, templates, THRESHOLD)
    ms = (time.time() - t0) / 5 * 1000
    print(f"Durée d'une observation 1280x720 avec {len(templates)} images : {ms:.0f} ms")

    print("\nRÉSULTAT :", "tous les tests réussis" if not failures else f"{failures} échec(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
