#!/bin/bash
set -e

# Setup environment
source /venv/main/bin/activate
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install nerfstudio

# Compile gsplat
MAX_JOBS=3 python -c "from gsplat.cuda._backend import _C; print('gsplat built')"

# Unzip data
unzip -q upload.zip -d /root/data/
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

# Render PPMs
echo "Rendering PPMs..."
./renderer /root/exports/20260718_230045/splat.ply /root/exports/20260718_230045/render.ppm
./renderer /root/exports/20260718_230238/splat.ply /root/exports/20260718_230238/render.ppm
./renderer /root/data/iceland_splat.ply /root/exports/iceland_render.ppm

echo "Done!"
