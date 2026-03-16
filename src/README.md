# src/ — Codice sorgente
**Ultimo aggiornamento:** 2026-03-16

## Struttura

```
src/
├── patch_on_plate/           ← Primo esperimento: texture su targa + confronto YOLO
│   └── patch_on_plate.py     ← Script principale
├── vehicle_counting_model/   ← YOLOv8n pre-trainato per vehicle counting
│   ├── yolov8n.pt            ← Pesi del modello
│   └── ...                   ← Notebook e dati del modello originale
└── README.md                 ← Questo file
```

## patch_on_plate

Sovrappone la texture CARLA (T_LicensePlate_d.TGA) sulle targhe del dataset CCPD usando perspective warp, poi confronta le detection di YOLOv8.

```bash
# Dal root del progetto, con il venv attivo
source .venv/bin/activate
python src/patch_on_plate/patch_on_plate.py --n_images 50
```

Opzioni:
- `--n_images`: numero di immagini da processare (default: 50)
- `--texture`: path alla texture (default: T_LicensePlate_d.TGA)
- `--weights`: path ai pesi YOLO (default: src/vehicle_counting_model/yolov8n.pt)
- `--dataset`: path alla directory delle immagini CCPD (default: data/CCPD2020/ccpd_green/train)
- `--output_dir`: directory di output (default: experiments/patch_on_plate)

Output:
- `results.csv`: dettagli per ogni detection (confidence originale vs. con patch)
- `summary.json`: metriche aggregate
- `comparisons/`: immagini side-by-side con bounding box

## vehicle_counting_model

Modello YOLOv8n pre-trainato (COCO, 80 classi). Usato per la detection dei veicoli.
Ha un suo venv separato in `.venv/`.

## Dipendenze

Tutte le dipendenze sono nel venv di progetto (`.venv/` nella root):
- opencv-python-headless
- ultralytics (YOLOv8)
- Pillow
- numpy
- matplotlib
