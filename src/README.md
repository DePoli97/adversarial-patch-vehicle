# src/ — Codice sorgente
**Ultimo aggiornamento:** 2026-04-29

## Struttura

```
src/
├── patch_on_surface/         ← Esperimento: texture su superficie veicolo + confronto YOLO
│   ├── patch_on_surface.py   ← Script principale
│   └── adversarial_patch_lab.ipynb  ← Notebook training patch (rear-window 1024×512)
├── carla_scenario/           ← Scenario CARLA two-vehicles (Nissan Micra leader)
├── vehicle_counting_model/   ← YOLOv8n pre-trainato per vehicle counting
│   ├── yolov8n.pt
│   └── ...
└── README.md
```

## patch_on_surface

Sovrappone la texture CARLA sulle targhe del dataset CCPD usando perspective warp, poi confronta le detection di YOLOv8. Output di default: `experiments/patch_on_rear_window/`.

```bash
# Dal root del progetto, con il venv attivo
source .venv/bin/activate
python src/patch_on_surface/patch_on_surface.py --n_images 50
```

Opzioni:
- `--n_images`: numero di immagini da processare (default: 50)
- `--texture`: path alla texture (default: T_LicensePlate_d.TGA)
- `--weights`: path ai pesi YOLO (default: src/vehicle_counting_model/yolov8n.pt)
- `--dataset`: path alla directory delle immagini CCPD (default: data/CCPD2020/ccpd_green/train)

Output:
- `results.csv`: dettagli per ogni detection (confidence originale vs. con patch)
- `summary.json`: metriche aggregate
- `comparisons/`: immagini side-by-side con bounding box

## vehicle_counting_model

Modello YOLOv8n pre-trainato (COCO, 80 classi). Usato per la detection dei veicoli.

## Dipendenze

Tutte le dipendenze sono nel venv di progetto (`.venv/` nella root):
- opencv-python-headless
- ultralytics (YOLOv8)
- Pillow
- numpy
- matplotlib
