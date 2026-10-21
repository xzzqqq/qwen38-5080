"""Runner package：模型、op、loader、spec。"""

from runner.env import configure_cuda_allocator

# 必须在第一次 CUDA 分配之前设置；只改 allocator 策略，不改数值。
configure_cuda_allocator()
