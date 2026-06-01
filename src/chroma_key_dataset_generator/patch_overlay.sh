#!/bin/bash

cd ../..

python3 src/chroma_key_dataset_generator/overlay_on_tga.py \
  --base assets/chroma_key/Vh_Truck_CarlaCola_BodyworkMat_BaseColor.TGA \
  --overlay assets/chroma_key/yellow_marker.TGA \
  --x 3000 --y 3000 \
  --scale 1.4 \
  --rotate 0 \
  --output assets/chroma_key/carlacola_with_yellow.TGA
