#!/bin/bash

cd ../..

python3 src/chroma_key_dataset_generator/overlay_on_tga.py \
  --base assets/chroma_key/Vh_Truck_CarlaCola_BodyworkMat_BaseColor.TGA \
  --overlay assets/chroma_key/yellow_marker.TGA \
  --x 2900 --y 2400 \
  --scale 0.8 \
  --rotate 0 \
  --output assets/chroma_key/carlacola_with_yellow.TGA
