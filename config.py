"""Réglages de l'application, stockés dans config.json à côté du .exe."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    window_title_contains: str = "BlueStacks"   # partie du titre de la fenêtre à observer
    threshold: float = 0.80                     # confiance minimale (0.50 à 0.99)
    scan_interval_ms: int = 500                 # pause entre deux observations
    cooldown_after_click_s: float = 4.0         # pause après un clic (laisse le jeu réagir)
    save_diagnostics: bool = True               # enregistre une image annotée à chaque clic


def load_config(path: Path) -> Config:
    cfg = Config()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)
    except (FileNotFoundError, ValueError):
        pass
    return cfg


def save_config(cfg: Config, path: Path) -> None:
    path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
