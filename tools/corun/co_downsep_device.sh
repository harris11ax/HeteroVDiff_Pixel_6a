#!/system/bin/sh
# P3.7j — `ffn.down` chunked-K TPU path (downsep) under GPU load.
#
# The exact recipe's last solo term. `npu_block downsep` runs 7 x FC[1536,1280]
# + 6 x ADD per 384-token tile, which is exactly the shape the 0.2580 s/block
# reference came from (RESULTS_npu_tn.md §4) — same binary, same synthetic
# activation. Here it runs (a) alone, (b) after a solo exact-recipe arm that
# pins T_N's other term to *this* boot and session, and (c) while the Mali
# Preset-A block executes. Every tile stamps CLOCK_MONOTONIC, so "it really
# overlapped" is computed from the artifact.
#
#   $1 = GPU solo iterations           $6 = GPU GEMM variant
#   $2 = GPU iterations in the co-run  $7 = downsep tiles per block
#   $3 = downsep solo iterations       $8 = downsep M (rows per tile)
#   $4 = downsep co-run iterations     $9 = seconds before the GPU joins
#   $5 = GPU M (tokens)
SOLO=$1; COIT=$2; NSOLO=$3; NCO=$4; MG=$5; VAR=$6; TILES=$7; MD=$8; DELAY=$9
# `${10}`, not `$10` — in POSIX sh `$10` is `${1}0`, which turned "--recipe-reps 8" into
# 100 passes (measured 2026-11-16: reps=100 in the SUMMARY line, 74 s instead of 15 s).
RECIPE=${10}
CL=/data/local/tmp/gemm8.cl
BLK=/data/local/tmp/hv_blk.bin
cd /data/local/tmp || exit 1
rm -f co_down_gpu_solo.txt co_down_gpu.txt co_down_npu_solo.txt co_down_npu.txt co_down_recipe.txt

echo "PHASE|solo_gpu|start=$(date +%s%N)"
./gemm_bench block "$CL" "$VAR" "$SOLO" "$MG" > co_down_gpu_solo.txt 2>&1
echo "PHASE|solo_gpu|end=$(date +%s%N)"

# same-session, same-boot reference for the recipe arm of T_N (0 if disabled)
if [ "$RECIPE" -gt 0 ]; then
  echo "PHASE|recipe_solo|start=$(date +%s%N)"
  ./hv_proc_b --block "$BLK" --recipe 1 --recipe-reps "$RECIPE" \
    --recipe-row 192 --recipe-col 384 --recipe-tokens 1536 \
    --down skip --decode 0 > co_down_recipe.txt 2>&1
  echo "PHASE|recipe_solo|end=$(date +%s%N)"
fi

echo "PHASE|downsep_solo|start=$(date +%s%N)"
./npu_block downsep "$TILES" "$NSOLO" "$MD" > co_down_npu_solo.txt 2>&1
echo "PHASE|downsep_solo|end=$(date +%s%N)"

echo "PHASE|downsep_launch|start=$(date +%s%N)"
./npu_block downsep "$TILES" "$NCO" "$MD" > co_down_npu.txt 2>&1 &
NPUPID=$!

sleep "$DELAY"

echo "PHASE|co_gpu|start=$(date +%s%N)"
./gemm_bench block "$CL" "$VAR" "$COIT" "$MG" > co_down_gpu.txt 2>&1
echo "PHASE|co_gpu|end=$(date +%s%N)"

wait $NPUPID
echo "PHASE|downsep_done|end=$(date +%s%N)"
