#!/bin/bash
set -e

source /venv/main/bin/activate

# Extract data properly handling Windows backslashes
mkdir -p /root/data
python -c "
import zipfile
import os

with zipfile.ZipFile('/root/upload.zip', 'r') as z:
    for info in z.infolist():
        info.filename = info.filename.replace('\\\\', '/')
        z.extract(info, '/root/data/')
"
cd /root/data

# Train first scene
echo "Training 20260718_230045..."
ns-train splatfacto --data 20260718_230045 --max-num-iterations 7000 --vis tensorboard colmap --colmap-path sparse/0 --images-path images
CFG1=$(ls -t /root/outputs/20260718_230045/splatfacto/*/config.yml | head -1)
ns-export gaussian-splat --load-config "$CFG1" --output-dir /root/exports/20260718_230045

# Train second scene
echo "Training 20260718_230238..."
ns-train splatfacto --data 20260718_230238 --max-num-iterations 7000 --vis tensorboard colmap --colmap-path sparse/0 --images-path images
CFG2=$(ls -t /root/outputs/20260718_230238/splatfacto/*/config.yml | head -1)
ns-export gaussian-splat --load-config "$CFG2" --output-dir /root/exports/20260718_230238

# Compile CUDA renderer
echo "Compiling renderer-cuda..."
cd /root/data/renderer-cuda
nvcc -O3 -use_fast_math forward.cu main.cpp -o renderer
chmod +x renderer

# Render PPMs
echo "Rendering PPMs..."
./renderer /root/exports/20260718_230045/splat.ply /root/exports/20260718_230045/render.ppm
./renderer /root/exports/20260718_230238/splat.ply /root/exports/20260718_230238/render.ppm
./renderer /root/data/iceland_splat.ply /root/exports/iceland_render.ppm

echo "Done!"
