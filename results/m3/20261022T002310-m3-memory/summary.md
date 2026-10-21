M3 分阶段显存
run_dir: /home/xzq/workspace/qwen/results/m3/20261022T002310-m3-memory
steps: 511

| scenario | execution | depth | capacity | load peak GB | cache init GB | prefill peak GB | capture pool GB | prefill→warmup→capture→prefill GB | graph-resident prefill peak GB | decode peak GB | max peak GB | min free GB | reset->prefill delta GB | KV ptr stable | stale state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eager-8192 | eager | 7680 | dynamic | 14.458 | 14.452 | 15.620 | - | - | 15.621 | 15.428 | 15.621 | 0.411 | 0.0 MB | True | True |
| exact-8192 | static-exact | 7680 | 8192 | 14.458 | 14.989 | 15.654 | - | - | 15.654 | 15.428 | 15.655 | 0.512 | 0.0 MB | True | True |
| cap-8192 | static-cap | 7680 | 8192 | 14.458 | 14.989 | 15.654 | - | - | 15.654 | 15.428 | 15.655 | 0.512 | 0.0 MB | True | True |
| graph-8192 | graph | 7680 | 8192 | 14.458 | 14.989 | 15.654 | 0.034 | 15.654 / 15.458 / 15.491 / 15.722 | 15.722 | 15.293 | 15.722 | 0.119 | 0.0 MB | True | True |
| cap-512-1024 | static-cap | 512 | 1024 | 14.458 | 14.519 | 14.810 | - | - | 14.811 | 14.781 | 14.811 | 1.374 | 0.0 MB | True | True |
| graph-512-1024 | graph | 512 | 1024 | 14.458 | 14.519 | 14.810 | 0.034 | 14.811 / 14.811 / 14.843 / 14.879 | 14.879 | 14.823 | 14.879 | 1.170 | 0.0 MB | True | True |

