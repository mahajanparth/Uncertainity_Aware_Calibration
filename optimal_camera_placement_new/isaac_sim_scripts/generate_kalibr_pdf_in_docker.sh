#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-kalibr}"
KALIBR_CREATE_TARGET="${2:-kalibr/aslam_offline_calibration/kalibr/python/kalibr_create_target_pdf}"
OUT_DIR="/workspace/isaac_sim_scripts/generated_assets"
OUT_NAME="aprilgrid_6x6_kalibr_generated.pdf"
HOST_LOG="isaac_sim_scripts/generated_assets/kalibr_pdf_docker.log"

mkdir -p isaac_sim_scripts/generated_assets
exec > >(tee "$HOST_LOG") 2>&1
set -x

echo "[kalibr-pdf host] started: $(date -Is)"
echo "[kalibr-pdf host] cwd: $PWD"
echo "[kalibr-pdf host] image: $IMAGE"
echo "[kalibr-pdf host] target generator: $KALIBR_CREATE_TARGET"
echo "[kalibr-pdf host] output: isaac_sim_scripts/generated_assets/$OUT_NAME"
echo "[kalibr-pdf host] docker:"
command -v docker
docker --version

docker run --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc "
    set -eo pipefail
    echo '[kalibr-pdf] image: $IMAGE'
    echo '[kalibr-pdf] workspace:'
    pwd
    echo '[kalibr-pdf] checking setup files'
    ls -d /catkin_ws/devel/setup.bash /kalibr_ws/devel/setup.bash /root/catkin_ws/devel/setup.bash 2>/dev/null || true
    set +u
    source /catkin_ws/devel/setup.bash 2>/dev/null || \
      source /kalibr_ws/devel/setup.bash 2>/dev/null || \
      source /root/catkin_ws/devel/setup.bash 2>/dev/null || true
    set -u
    echo '[kalibr-pdf] setup sourcing finished'
    echo '[kalibr-pdf] kalibr_create_target_pdf location:'
    command -v kalibr_create_target_pdf || true
    find / -path '*kalibr_create_target_pdf' -type f 2>/dev/null | head -n 20 || true

    mkdir -p '$OUT_DIR/kalibr_pdf_target'
    cd '$OUT_DIR/kalibr_pdf_target'
    rm -f ./*.pdf
    generator='$KALIBR_CREATE_TARGET'
    if [ -f \"\$generator\" ]; then
      generator=\"/workspace/\$generator\"
    elif [ -f \"/workspace/\$generator\" ]; then
      generator=\"/workspace/\$generator\"
    elif command -v kalibr_create_target_pdf >/dev/null 2>&1; then
      generator=\"\$(command -v kalibr_create_target_pdf)\"
    else
      generator=\"\$(find /workspace / -path '*kalibr_create_target_pdf' -type f 2>/dev/null | head -n 1 || true)\"
    fi
    if [ -z \"\$generator\" ] || [ ! -f \"\$generator\" ]; then
      echo '[kalibr-pdf] ERROR: kalibr_create_target_pdf not found inside container or mounted workspace' >&2
      exit 127
    fi
    echo '[kalibr-pdf] using generator:'
    echo \"\$generator\"
    echo '[kalibr-pdf] generating in:'
    pwd
    python3 \"\$generator\" --type apriltag --nx 6 --ny 6 --tsize 0.088 --tspace 0.3
    echo '[kalibr-pdf] files after generation:'
    find . -maxdepth 2 -type f -print | sort
    pdf=\$(find . -maxdepth 2 -type f -name '*.pdf' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [ -z \"\$pdf\" ]; then
      echo '[kalibr-pdf] ERROR: no PDF was generated' >&2
      exit 1
    fi
    cp \"\$pdf\" '$OUT_DIR/$OUT_NAME'
    chmod 664 '$OUT_DIR/$OUT_NAME' || true
    ls -lh '$OUT_DIR/$OUT_NAME'
  "

echo "[kalibr-pdf host] done: $(date -Is)"
