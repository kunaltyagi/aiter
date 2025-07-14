set -vex

git checkout kt/bench
git log --oneline -n1
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp8 2>&1 >softmax_prefill_a.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp16 2>&1 >softmax_prefill_b.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp32 2>&1 >softmax_prefill_c.txt

git checkout HEAD^
git log --oneline -n1
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp8 2>&1 >fma_prefill_a.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp16 2>&1 >fma_prefill_b.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp32 2>&1 >fma_prefill_c.txt

git checkout HEAD^
git log --oneline -n1
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp8 2>&1 >exp_prefill_a.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp16 2>&1 >exp_prefill_b.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp32 2>&1 >exp_prefill_c.txt

git checkout HEAD^
git log --oneline -n1
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp8 2>&1 >exp2_prefill_a.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp16 2>&1 >exp2_prefill_b.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp32 2>&1 >exp2_prefill_c.txt

git checkout HEAD^
git log --oneline -n1
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp8 2>&1 >baseline_prefill_a.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp16 2>&1 >baseline_prefill_b.txt
HIP_VISIBLE_DEVICES=3 python op_tests/op_benchmarks/triton/bench_mha.py -dtype fp32 2>&1 >baseline_prefill_c.txt

