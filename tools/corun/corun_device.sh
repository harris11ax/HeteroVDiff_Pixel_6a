#!/system/bin/sh
# Co-run T_G: Mali (OpenCL) and Edge TPU (NNAPI) driven at the same time.
# Pushed to /data/local/tmp/corun_device.sh and run by tools/corun/co_run.py.
#
#   $1 = GPU solo iterations (block mode, M tokens)
#   $2 = NPU corun rounds (one round = 9 GEMMs x `tiles` executes)
#   $3 = GPU iterations during the co-run window
#   $4 = tiles per GEMM per round
#   $5 = M on the GPU (tokens per block)
#   $6 = M on the NPU (per-graph token ceiling)
#   $7 = GEMM variant name for the GPU
#
# Everything lands in /data/local/tmp/co_*.txt; the host pulls and analyses them.
# The NPU process is started FIRST so its 9 compilations are finished and the
# TPU is already executing before the GPU joins — that leaves genuine solo NPU
# samples inside the very same process, which is what makes the contended-vs-solo
# comparison an A/B on one code path rather than a cross-capture comparison.
SOLO=$1; ROUNDS=$2; COIT=$3; TILES=$4; MG=$5; MN=$6; VAR=$7
CL=/data/local/tmp/gemm8.cl
cd /data/local/tmp || exit 1
rm -f co_gpu_solo.txt co_gpu.txt co_npu.txt

echo "PHASE|solo_gpu|start=$(date +%s%N)"
./gemm_bench block "$CL" "$VAR" "$SOLO" "$MG" > co_gpu_solo.txt 2>&1
echo "PHASE|solo_gpu|end=$(date +%s%N)"

./npu_block corun "$TILES" "$ROUNDS" "$MN" 0 -1 > co_npu.txt 2>&1 &
NPUPID=$!
i=0
while [ $i -lt 240 ]; do
  grep -q CORUN_READY co_npu.txt && break
  sleep 0.25
  i=$((i+1))
done
echo "PHASE|npu_ready|polls=$i|start=$(date +%s%N)"
# A short solo window: ~1.5 s of pure TPU work before the GPU joins.
sleep 1.5

echo "PHASE|co_gpu|start=$(date +%s%N)"
./gemm_bench block "$CL" "$VAR" "$COIT" "$MG" > co_gpu.txt 2>&1
echo "PHASE|co_gpu|end=$(date +%s%N)"

wait $NPUPID
echo "PHASE|corun|done=$(date +%s%N)"
