#!/system/bin/sh
# P3.7d — co-run treatment of the *exact* NPU recipe (Mali + Edge TPU at once).
# Pushed to /data/local/tmp/co_recipe_device.sh and run by tools/corun/co_recipe.py.
#
#   $1 = GPU solo iterations (block mode)
#   $2 = GPU iterations while the recipe runs
#   $3 = recipe reps (passes over the 9 GEMMs)
#   $4 = M on the GPU (tokens per block)
#   $5 = GEMM variant name for the GPU
#   $6 = recipe row tile
#   $7 = recipe column block
#   $8 = recipe tokens per block
#   $9 = seconds to wait after launching the recipe before the GPU joins
#
# Both processes stamp CLOCK_MONOTONIC (gemm_bench: TS|gpu t0/t1; proc_b: RECIPE|col
# t0/t1), so "they really ran at the same time" is computed from the artifact, not
# claimed. Everything lands in /data/local/tmp/co_*.txt.
SOLO=$1; COIT=$2; REPS=$3; MG=$4; VAR=$5; ROW=$6; COL=$7; TOK=$8; DELAY=$9
CL=/data/local/tmp/gemm8.cl
BLK=/data/local/tmp/hv_blk.bin
cd /data/local/tmp || exit 1
rm -f co_gpu_solo.txt co_gpu.txt co_recipe.txt

echo "PHASE|solo_gpu|start=$(date +%s%N)"
./gemm_bench block "$CL" "$VAR" "$SOLO" "$MG" > co_gpu_solo.txt 2>&1
echo "PHASE|solo_gpu|end=$(date +%s%N)"

echo "PHASE|recipe_launch|start=$(date +%s%N)"
./hv_proc_b --block "$BLK" --recipe 1 --recipe-reps "$REPS" \
  --recipe-row "$ROW" --recipe-col "$COL" --recipe-tokens "$TOK" \
  --down skip --decode 0 > co_recipe.txt 2>&1 &
NPUPID=$!

sleep "$DELAY"

echo "PHASE|co_gpu|start=$(date +%s%N)"
./gemm_bench block "$CL" "$VAR" "$COIT" "$MG" > co_gpu.txt 2>&1
echo "PHASE|co_gpu|end=$(date +%s%N)"

wait $NPUPID
echo "PHASE|recipe_done|end=$(date +%s%N)"
