# Mirage MPK 框架深度技术分析报告

> 基于代码仓库分析与 OSDI'25 论文整理  
> 分析日期：2026-06-09

---

## 目录

1. [整体架构与核心设计思想](#1-整体架构与核心设计思想)
2. [编译流水线：从模型定义到 Megakernel](#2-编译流水线从模型定义到-megakernel)
3. [运行时系统与事件驱动执行](#3-运行时系统与事件驱动执行)
4. [核心任务实现与推理优化技术](#4-核心任务实现与推理优化技术)
5. [模型支持：Qwen3 实现详解](#5-模型支持qwen3-实现详解)
6. [性能优化机制](#6-性能优化机制)
7. [新模型接入完整指南](#7-新模型接入完整指南)

---

## 1. 整体架构与核心设计思想

### 1.1 什么是 Megakernel

MPK（Mirage Persistent Kernel）是 Mirage 框架中用于大规模 LLM 推理的**持久化内核框架**。其核心创新是 **Megakernel** 概念——将整个 LLM 推理流程（包括多层模型计算、注意力机制、解码等）编译为**单个持久的 CUDA 内核**。

与传统推理框架的根本区别：

| 维度 | 传统方式（PyTorch/vLLM） | MPK Megakernel |
|------|------------------------|----------------|
| 内核粒度 | 每个操作一个内核（linear、attention 等） | 整个推理迭代一个内核 |
| 内核启动 | 频繁，每步数十次 CPU→GPU 调用 | 一次启动，持续运行 |
| 同步开销 | 大量 cudaStreamSynchronize | 内部原子事件，无 CPU 同步 |
| SM 状态 | 频繁进出低功耗 | 始终热就绪 |
| 批次调度 | CPU 控制，有延迟 | GPU 内部自主调度 |

### 1.2 系统整体架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户 Python 代码                        │
│   mpk = MPK(config) → mpk.build() → mpk.compile() → mpk()│
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Python API 层                               │
│  MPK(mpk.py) → PersistentKernel(persistent_kernel.py)   │
│  各种 layer 方法：rmsnorm_layer, linear_layer, ...        │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│            计算图层（KNGraph + TBGraph）                  │
│  KNGraph：设备级张量与算子图（C++ CyKNGraph）             │
│  TBGraph：线程块级计算图（C++ CyTBGraph）                 │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│               任务图生成 & 代码生成                        │
│  generate_task_graph() → 任务图 JSON + CUDA 源码         │
└────────────────┬────────────────────────────────────────┘
                 │ nvcc 编译
┌────────────────▼────────────────────────────────────────┐
│            编译产物（.so 共享库）                          │
│  init_persistent_kernel() + launch_persistent_kernel()  │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│         GPU 运行时（Megakernel 执行）                     │
│  Worker SM（计算）  ←→  Scheduler SM（调度）              │
│  事件驱动任务队列，persistent loop 执行所有层             │
└─────────────────────────────────────────────────────────┘
```

### 1.3 主要目录结构

```
mirage/
├── python/mirage/
│   ├── kernel.py                    # KNGraph 高层 API
│   └── mpk/
│       ├── mpk.py                   # MPK 主入口类 + MPKMetadata
│       ├── persistent_kernel.py     # PersistentKernel：layer 方法 + 编译
│       ├── model_registry.py        # 模型注册机制
│       ├── multigpu.py              # 多 GPU AllReduce 策略
│       └── models/
│           ├── graph_builder.py     # GraphBuilder 抽象基类 + MirageModelConfig
│           ├── qwen3/builder.py     # Qwen3 模型实现
│           └── deepseek_v3/builder.py  # DeepSeek V3 模型实现
├── include/mirage/persistent_kernel/
│   ├── runtime_header.h             # 运行时数据结构定义
│   ├── persistent_kernel.cuh        # Worker/Scheduler 内核实现
│   └── tasks/
│       ├── hopper/                  # Hopper(SM90) 任务实现
│       ├── blackwell/               # Blackwell(SM100) 任务实现
│       └── ampere/                  # Ampere(SM80) 任务实现
└── demo/                            # 使用示例
```

---

## 2. 编译流水线：从模型定义到 Megakernel

### 2.1 整体编译流程（七步走）

```
Step 1: 用户定义模型配置（MPKMetadata）
        ↓
Step 2: 模型构建器注册张量并调用 layer 方法（build()）
        ↓
Step 3: 构建 KNGraph + TBGraph 计算图
        ↓
Step 4: 任务图生成（generate_task_graph()）→ JSON + CUDA 代码
        ↓
Step 5: nvcc 编译生成 .so 共享库（compile()）
        ↓
Step 6: Python 动态加载 .so，调用 init_persistent_kernel()
        ↓
Step 7: 调用 launch_persistent_kernel() 执行推理
```

### 2.2 Step 1-2：Python API 层

**MPK 主类**（[python/mirage/mpk/mpk.py:125](python/mirage/mpk/mpk.py#L125)）：

```python
class MPK:
    def __init__(self, meta: MPKMetadata):
        self.persistent_kernel = PersistentKernel(
            mode=args.mode,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_schedulers,
            max_num_batched_tokens=self.max_num_batched_tokens,
            ...
        )

    def build(self):
        model_builder_class = get_builder(self.model_name)
        self.model_builder = model_builder_class(self.persistent_kernel)
        self.model_builder.build_from_model(...)  # 或 build_from_config()

    def generate_task_graph(self):
        results = self.persistent_kernel.kn_graph.generate_task_graph(
            num_gpus=self.world_size, my_gpu_id=self.rank
        )

    def compile(self, output_dir=None):
        self.persistent_kernel.compile(output_dir=output_dir, ...)
```

**MPKMetadata 关键参数**（[python/mirage/mpk/mpk.py:16](python/mirage/mpk/mpk.py#L16)）：

```python
@dataclass
class MPKMetadata:
    mode: str = "offline"              # offline/online/online_pinned
    max_seq_length: int = 0
    max_num_batched_requests: int = 0
    max_num_batched_tokens: int = 0
    max_num_pages: int = 0
    page_size: int = 0
    max_sm_num: int = 108              # H100=108, B200=148
    model_name: Optional[str] = None
    world_size: int = 1                # 张量并行度
    tokens: Optional[torch.Tensor] = None
    qo_indptr_buffer: Optional[torch.Tensor] = None
    paged_kv_indices_buffer: Optional[torch.Tensor] = None
    # ... paged KV cache 元数据张量
```

### 2.3 Step 3：KNGraph 与 TBGraph 计算图构建

**KNGraph（Kernel Graph，设备级图）**（[python/mirage/kernel.py:208](python/mirage/kernel.py#L208)）：

KNGraph 是设备内存级别的计算图，包含 DTensor（Device Tensor）和算子节点。

```python
class KNGraph:
    def new_input(self, dims, strides=None, dtype=float16) -> DTensor:
        # 创建设备张量，绑定到 GPU 内存
    
    def attach_torch_tensor(self, dtensor, torch_tensor, name):
        # 将 PyTorch 张量绑定到图节点（raw pointer）
    
    def customized(self, inputs, bgraph: TBGraph) -> list[DTensor]:
        # 注册自定义 TB 级子图
    
    def register_task(self, bgraph, task_type: str, params=[]):
        # 将 TB 图绑定到特定任务实现
```

**TBGraph（Threadblock Graph，线程块级图）**：

每个 layer 方法都会创建一个 TBGraph，描述该 layer 的并行模式：

```python
# 以 linear_layer 为例（persistent_kernel.py:1737-1766）
def linear_layer(self, input, weight, output, grid_dim, block_dim):
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    
    # new_input(tensor, partition_map, store_dim, in_dmem)
    # partition_map: (-1,-1,-1)=全局，(0,-1,-1)=按grid.x分partition dim0
    tb_graph.new_input(input,  (-1, -1, -1), 1, True)   # 全局读
    tb_graph.new_input(weight, (0, -1, -1), 1, True)    # grid.x 分 dim0
    tb_graph.new_input(output, (1, -1, -1), -1, True)   # grid.x 分 dim1
    
    self.kn_graph.customized([input, weight, output], tb_graph)
    
    # 根据目标 GPU 架构选择最优实现
    if self.target_cc >= 100:
        self.kn_graph.register_task(tb_graph, "linear_sm100")
    elif self.target_cc >= 90:
        self.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
    elif self.target_cc >= 80:
        self.kn_graph.register_task(tb_graph, "linear")
```

**partition_map 说明**：

| partition_map | 含义 |
|--------------|------|
| `(-1, -1, -1)` | 全局访问，所有线程块读同一份数据 |
| `(0, -1, -1)` | 按 grid.x 分割第 0 维 |
| `(1, -1, -1)` | 按 grid.x 分割第 1 维 |
| `(0, 1, -1)` | grid.x 分第 0 维，grid.y 分第 1 维 |
| `(0, 2, -1)` | grid.x 分第 0 维，grid.y 分第 2 维 |

### 2.4 Step 4：任务图生成

`generate_task_graph()` 调用 C++ 侧的图分析与代码生成：

```python
# mpk.py:469-476
def generate_task_graph(self):
    results = self.persistent_kernel.kn_graph.generate_task_graph(
        num_gpus=self.world_size, my_gpu_id=self.rank
    )
    # results 包含：
    # - results["cuda_code"]：生成的完整 CUDA 源码
    # - results["json_file"]：任务图 JSON 序列化
```

**任务图 JSON 结构示意**：

```json
{
  "tasks": [
    {
      "task_id": 0,
      "task_type": "embedding",
      "input_tensor_ids": [0, 1],
      "output_tensor_ids": [2],
      "grid_dim": [1, 1, 1],
      "block_dim": [128, 1, 1]
    },
    {
      "task_id": 1,
      "task_type": "linear_sm100",
      "input_tensor_ids": [2, 3],
      "output_tensor_ids": [4],
      "grid_dim": [32, 1, 1],
      "block_dim": [128, 1, 1]
    }
  ],
  "events": [
    {"event_id": 0, "type": "LAUNCH_TASKS", "tasks": [0]},
    {"event_id": 1, "type": "LAUNCH_DEPENDENT_TASKS", "tasks": [1], "depends_on": [0]}
  ]
}
```

### 2.5 Step 5：nvcc 编译

`get_compile_command()` 函数（[persistent_kernel.py:184](python/mirage/mpk/persistent_kernel.py#L184)）生成完整 nvcc 命令：

```python
def get_compile_command(mpk, target_cc, cc, file_name, ...):
    common_cmd = [
        "nvcc",
        file_name,
        "-O3",
        "-lineinfo",
        f"-I{mirage_inc_path}",
        f"-I{cutlass_include}",
        f"-I{json_include}",
        f"-DMAX_WORKER_PER_SCHEDULER={max_worker_per_scheduler}",
        "-shared",
        "-std=c++17",
        "-use_fast_math",
        "-lcuda", "-lcudart",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-o", py_so_path,
    ]

    # 架构特定参数
    if target_cc == 90:    # Hopper H100
        arch_cmd = [
            "-gencode=arch=compute_90a,code=sm_90a",
            "-DMPK_ENABLE_TMA",          # 启用 TMA
            "-DMIRAGE_GRACE_HOPPER",
        ]
    elif target_cc == 100: # Blackwell B200
        arch_cmd = [
            "-gencode=arch=compute_100a,code=sm_100a",
            "-DMPK_ENABLE_TMA",
            "-DMIRAGE_GRACE_BLACKWELL",
        ]

    # 推理模式宏（编译时常量）
    mode_flags = [f"-DMODE_{mpk.mode.upper()}"]

    # 批处理配置（编译时常量，影响 shared memory 分配）
    config_flags = [
        f"-DMPK_TARGET_CC={target_cc}",
        f"-DMPK_MAX_NUM_BATCHED_REQUESTS={mpk.max_num_batched_requests}",
        f"-DMPK_MAX_NUM_BATCHED_TOKENS={mpk.max_num_batched_tokens}",
        f"-DMPK_MAX_NUM_PAGES={mpk.max_num_pages}",
        f"-DMPK_PAGE_SIZE={mpk.page_size}",
        f"-DMPK_MAX_SEQ_LENGTH={mpk.max_seq_length}",
    ]
```

**关键编译时宏定义**：

| 宏定义 | 作用 |
|--------|------|
| `MIRAGE_GRACE_HOPPER` | 启用 Hopper SM90 特化代码路径 |
| `MIRAGE_GRACE_BLACKWELL` | 启用 Blackwell SM100 特化代码路径 |
| `MPK_ENABLE_TMA` | 启用 Tensor Memory Accelerator |
| `MODE_OFFLINE/ONLINE/ONLINE_PINNED` | 推理服务模式 |
| `MPK_MAX_NUM_BATCHED_REQUESTS` | 最大并发请求数（影响 KV cache 分配） |
| `MPK_MAX_NUM_PAGES` | KV cache 最大页数 |
| `MPK_PAGE_SIZE` | 每页 token 数 |
| `MIRAGE_USE_CUTLASS_KERNEL` | 使用 CUTLASS 高性能 GEMM |

### 2.6 Step 6-7：动态加载与运行

编译完成后，Python 动态加载 .so 并调用 C extension：

```python
# persistent_kernel.py:~2486
import importlib.util
spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

self.init_func = mod.init_func       # 初始化持久内核
self.launch_func = mod.launch_func   # 启动推理
self.finalize_func = mod.finalize_func

# HARD_CODE 中定义的 C Extension 接口（persistent_kernel.py:21-167）
# init_func → init_persistent_kernel()  （加载任务图到 GPU）
# launch_func → launch_persistent_kernel() （启动 megakernel）
```

**执行推理**（[python/mirage/mpk/mpk.py:486-498](python/mirage/mpk/mpk.py#L486)）：

```python
def __call__(self, **kwargs):
    if not self.is_compiled:
        self.compile()
    self.persistent_kernel(**kwargs)  # 调用 launch_func
```

---

## 3. 运行时系统与事件驱动执行

### 3.1 Persistent Kernel 执行模型

持久化内核启动后永不退出，形成两类并发执行的 SM 角色：

**Worker SM**（占总 SM 的 ~90%）：
- 从 worker_queue 取任务（TaskDesc）
- 等待 dependent_event 满足（自旋）
- 调用 `_execute_task()` 执行实际计算
- 完成后触发 trigger_event

**Scheduler SM**（占总 SM 的 ~10%）：
- 从 sched_queue 取事件（EventDesc）
- 根据 EventType 将任务分发给 Workers
- 管理批次准备（page 分配、请求调度）

**SM 分区配置**（[python/mirage/utils.py:33](python/mirage/utils.py#L33)）：

```python
def get_configurations_from_gpu(rank):
    sm_cnt = torch.cuda.get_device_properties(rank).multi_processor_count
    
    if sm_cnt >= 160:    worker = 144   # B200
    elif sm_cnt >= 132:  worker = 128   # H100
    elif sm_cnt >= 108:  worker = 96    # A100
    elif sm_cnt >= 68:   worker = 64    # A6000
    else:                worker = 20
    
    scheduler = 4 * (sm_cnt - worker)  # ~4:1 比例
    return worker, scheduler
```

### 3.2 数据结构：TaskId 与 EventId 编码

**TaskId（64 位）**（[include/mirage/persistent_kernel/runtime_header.h:85](include/mirage/persistent_kernel/runtime_header.h#L85)）：

```cpp
typedef unsigned long long int TaskId;

// 高 32 位：迭代次数（iteration_num）
// 低 32 位：任务在图中的位置索引
TaskId compute_task_id(iteration_num, position_index) {
    return (iteration_num << 32) | position_index;
}
```

**EventId（64 位）**：

```
[63:48] GPU ID (nvshmem)
[47:32] nvshmem tag
[31:0]  event 索引
```

**FullTaskDesc 结构**（[runtime_header.h:242](include/mirage/persistent_kernel/runtime_header.h#L242)）：

```cpp
struct FullTaskDesc {
    TaskType task_type;         // 任务类型枚举（见第4节）
    unsigned variant_id;
    int num_inputs, num_outputs;
    EventId trigger_event;      // 完成后触发的事件
    EventId dependent_event;    // 执行前等待的依赖事件
    TensorDesc inputs[7];       // 最多 7 个输入张量
    TensorDesc outputs[3];      // 最多 3 个输出张量
    union TaskMetadata {
        struct { int expert_offset; };         // MoE 路由
        struct {
            int16_t request_id;               // 分页注意力：请求 ID
            uint16_t kv_idx;                  // Split-KV 索引
            int merge_task_offset;            // 合并任务偏移
        };
        unsigned long long raw_payload;
    } task_metadata;
};
```

**TensorDesc 结构**（[runtime_header.h:220](include/mirage/persistent_kernel/runtime_header.h#L220)）：

```cpp
struct TensorDesc {
    int num_dims;
    void *base_ptr;
    void *tma_desc_ptrs[MAX_TMA_DESC_PER_TENSOR]; // TMA 描述符（Hopper/Blackwell）
    int data_type;
    int dim[MAX_TENSOR_DIMS];
    int stride[MAX_TENSOR_DIMS];
};
```

### 3.3 事件驱动执行模型

**EventType 枚举**（[runtime_header.h:210](include/mirage/persistent_kernel/runtime_header.h#L210)）：

```cpp
enum EventType {
    EVENT_LAUNCH_TASKS = 901,           // 启动独立任务批次
    EVENT_LAUNCH_MASSIVE_TASKS = 902,   // 启动大量任务（跨 Scheduler）
    EVENT_LAUNCH_DEPENDENT_TASKS = 903, // 启动有依赖的任务
    EVENT_END_OF_TASK_GRAPH = 910,      // 迭代完成，准备下一批次
    EVENT_TERMINATION = 911,            // 关闭内核
};
```

**依赖追踪机制**（EventCounter，[persistent_kernel.cuh:900](include/mirage/persistent_kernel/persistent_kernel.cuh#L900)）：

```cuda
// Worker 在执行任务前自旋等待依赖
if (task_desc->dependent_event != EVENT_INVALID_ID) {
    EventCounter needed = num_triggers[event_idx] * iteration_num;
    while (actual_counts < needed) {
        actual_counts = ld_acquire_sys_u64(&all_event_counters[event_idx]);
        __nanosleep(10);
    }
}
// 执行完成后，原子递增 trigger_event 计数器
atomicAdd(&all_event_counters[trigger_event_idx], 1);
```

### 3.4 批次管理：Paged KV Cache

**RuntimeConfig 中的批次元数据**（[runtime_header.h:320](include/mirage/persistent_kernel/runtime_header.h#L320)）：

```cpp
struct RuntimeConfig {
    int *step;                          // [total_requests]：各请求当前步数
    long long *input_tokens;            // [max_batched_tokens]：当前输入 token
    long long *output_tokens;           // [max_batched_tokens]：生成的 token
    int *qo_indptr_buffer;              // CSR 格式：Q/O 指针
    int *paged_kv_indptr_buffer;        // CSR 格式：KV 页面指针
    int *paged_kv_indices_buffer;       // 物理页面 ID 映射
    int *paged_kv_last_page_len_buffer; // 各请求最后一页有效长度
    int *page_queue;                    // 空闲页面池
    int *page_queue_head, *page_queue_tail;
};
```

### 3.5 Online_pinned 模式：零拷贝 CPU-GPU 通信

**锁自由环形缓冲区**（[runtime_header.h:369](include/mirage/persistent_kernel/runtime_header.h#L369)）：

```cpp
// 请求环（CPU 生产者 → GPU 消费者）
int32_t volatile *pinned_req_ready;     // 0=空, 1=就绪（acquire/release 语义）
int32_t *pinned_req_request_id;
int32_t *pinned_req_prompt_len;
int64_t *pinned_inbox_tokens;           // [ring_capacity * max_seq_len]

// 完成环（GPU 生产者 → CPU 消费者）
int32_t volatile *pinned_comp_ready;
int32_t *pinned_comp_request_id;
int32_t *pinned_comp_final_step;

// GPU 内部状态
int32_t *pinned_step;                   // 每步更新，CPU 可轮询实现流式输出
```

### 3.6 初始化/启动/终止流程

```python
# 1. 编译后初始化（mpk.py:414-427）
self.persistent_kernel.init_func(
    meta_tensors_ptr,          # 所有元数据张量指针
    profiler_buffer_ptr,
    mpi_rank, num_workers, num_local_schedulers,
    max_seq_length, total_num_requests, eos_token_id,
    model_tensor_names,        # 权重张量名称
    model_tensor_ptrs,         # 权重张量 GPU 指针
    json_path,                 # 任务图 JSON 路径
)

# 2. 每次推理前初始化请求资源
self.persistent_kernel.init_request_func()

# 3. 推理
self.persistent_kernel.launch_func(stream_ptr)
# GPU 侧：
#   prepare_kernel 重置队列
#   worker_kernel + scheduler_kernel 持续运行直到 TERMINATION 事件

# 4. 清理
self.persistent_kernel.finalize_func()
```

---

## 4. 核心任务实现与推理优化技术

### 4.1 TaskType 枚举：按架构分层的 100+ 任务类型

MPK 的所有计算单元通过 `TaskType` 枚举标识（[runtime_header.h:101](include/mirage/persistent_kernel/runtime_header.h#L101)），按 GPU 架构分四个区段：

```
[100-149]  通用任务（Ampere 兼容）
[150-198]  Hopper 专用任务（SM90，TASK_HOPPER_TASK_BEGIN ~ END）
[230-298]  Blackwell 专用任务（SM100，TASK_SM100_TASK_BEGIN ~ END）
[300-349]  多 GPU 任务（TASK_MULTIGPU_TASK_BEGIN ~ END）
[200-204]  调度器内部任务（TASK_SCHD_*）
```

**通用区（100-149，Ampere 兼容）**：

| TaskType | 值 | 说明 |
|----------|----|------|
| TASK_EMBEDDING | 101 | Token 嵌入查表 |
| TASK_RMS_NORM_LINEAR | 102 | RMSNorm + Linear 融合 |
| TASK_PAGED_ATTENTION_1/2 | 116/117 | 分页注意力两阶段（QK + PV） |
| TASK_SILU_MUL | 118 | SiLU × 门控融合激活 |
| TASK_LINEAR | 120 | 标准矩阵乘 |
| TASK_ARGMAX_PARTIAL | 110 | 并行 argmax 第一阶段 |
| TASK_ARGMAX_REDUCE | 111 | 并行 argmax 归约合并 |

**Hopper 区（150-198，SM90 专用）**：

| TaskType | 值 | 说明 |
|----------|----|------|
| TASK_LINEAR_SWAPAB_HOPPER | 155 | TMA + WGMMA，A/B 转置优化 |
| TASK_LINEAR_CUTLASS_HOPPER | 157 | 基于 CUTLASS 的 GEMM |
| TASK_PAGED_ATTENTION_HOPPER | 153 | TMA 加速分页注意力 |
| TASK_MOE_W13_LINEAR_SM90 | 161 | MoE gate+up 线性层 |
| TASK_SPLITK_LINEAR_SWAPAB_HOPPER | 163 | Split-K GEMM |

**Blackwell 区（230-298，SM100 专用）**：

| TaskType | 值 | 说明 |
|----------|----|------|
| TASK_LINEAR_SM100 | 253 | UMMA + TMEM 线性层 |
| TASK_SPLITK_LINEAR_SM100 | 251 | SM100 Split-K GEMM |
| TASK_ATTN_SM100 | 257 | 分页注意力 SM100 |
| TASK_MLA_DECODE_SM100 | 266 | DeepSeek MLA 解码 |
| TASK_MLA_REDUCE_SM100 | 267 | MLA 分布式合并 |
| TASK_LINEAR_FP8_SM100 | 276 | FP8 量化线性层 |
| TASK_MOE_W13_FP8_SM100 | 248 | MoE FP8 gate+up |
| TASK_QUANTIZE_FP8_SM100 | 275 | 激活 FP8 量化 |
| TASK_MTP_VERIFY_STRICT | 271 | 投机解码验证（严格贪心） |
| TASK_MTP_ACCEPT_COMMIT | 272 | 投机解码接受+提交 |

---

### 4.2 Hopper 硬件加速：TMA + WGMMA

**TMA（Tensor Memory Accelerator）** 是 Hopper（SM90）引入的异步 DMA 引擎，可在不占用计算 warp 的情况下将全局内存数据搬运到共享内存。

**实现文件**：[tasks/hopper/linear_swapAB_hopper.cuh](include/mirage/persistent_kernel/tasks/hopper/linear_swapAB_hopper.cuh)

核心实现模式（TMA + WGMMA 双 warpgroup 流水线）：

```cpp
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE, int Kstages>
__device__ void linear_swapAB_kernel_hopper(
    const TMA_A &tma_a, const TMA_B &tma_b, const TMA_OUT &tma_out) {

  // 1 Producer warpgroup（128 threads）：专负责 TMA 异步加载
  // 1 Consumer warpgroup（128 threads）：专负责 WGMMA 计算
  constexpr int CONSUMER_WARPGROUPS = 1;
  constexpr int PRODUCER_WARPGROUPS = 1;

  // TMA tile 与 swizzle 对齐
  constexpr int INPUT_TMA_TILE_SIZE = 64;
  constexpr int WEIGHT_TMA_TILE_SIZE = 64;

  // 流水线：Producer 异步加载下一个 K tile，Consumer 计算当前 K tile
  // tma::load_async(smem_buf, tma_desc, coord, barrier);
  // wgmma::mma_async_sync_aligned(acc, smem_a, smem_b);
}
```

"SwapAB" 的含义：将权重矩阵作为 A、激活作为 B（常规相反），使小 batch 推理时权重沿 K 维连续访问，显著提升 TMA 效率。`Kstages` 参数控制流水线深度（典型值 4~8），`BATCH_SIZE <= 16` 限制确保整个 M 维能填满单个 warpgroup 的 MMA 寄存器。

---

### 4.3 Blackwell 硬件加速：UMMA + TMEM

Blackwell（SM100）引入两个关键新特性：

**UMMA（Unified MMA / tcgen05）**：新一代矩阵乘指令，相比 Hopper WGMMA 理论算力提升 ~4×

**TMEM（Tensor Memory）**：SM100 专有的片上高速暂存内存，专门存放 MMA 累加器，避免频繁写回 shared memory

**实现文件**：[tasks/blackwell/linear_sm100_mpk.cuh](include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh)

```cpp
template <typename T_, typename TMA_A, typename TMA_B,
          int MMA_M, int MMA_N, int BATCH_SIZE, int OUTPUT_SIZE,
          int NUM_AB_STAGE = 8, int NUM_ACC_STAGE = 2, int NUM_C_STAGE = 4>
__device__ void linear_sm100_mpk_task_impl(
    const TMA_A &tma_a, const TMA_B &tma_b,
    BiasTensor mBias, const TMA_OUT &tma_out) {

  // CuTe SM100 UMMA tiled MMA
  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<T_, T_, float, MMA_M, MMA_N,
                                 cute::UMMA::Major::K,
                                 cute::UMMA::Major::K>{});

  // TMEM 分配：累加器放在 Tensor Memory，不占 shared memory 预算
  cute::arch::TmemAllocator tmem_alloc;
  auto tmem_acc = tmem_alloc.allocate<float>(MMA_N * NUM_ACC_STAGE);

  // 三级流水线：AB加载(8阶段) / 累加器切换(2阶段) / C写回(4阶段)
}
```

关键流水线参数：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `NUM_AB_STAGE` | AB 矩阵 TMA 加载流水线深度 | 8 |
| `NUM_ACC_STAGE` | TMEM 累加器双缓冲 | 2 |
| `NUM_C_STAGE` | 输出写回 TMA 阶段 | 4 |

---

### 4.4 算子融合

MPK 将多个算子融合成单个任务，消除中间结果写回 HBM 的开销：

**RMSNorm + Linear 融合**（`TASK_RMS_NORM_LINEAR = 102`）：
```
正常：  input → [RMSNorm] → norm_out(写HBM) → [Linear] → output
融合：  input → [RMSNorm+Linear] → output  （norm_out 仅在 shared memory 中）
```

**SiLU×MUL + Linear + 残差融合**（`TASK_SILU_MUL_LINEAR_WITH_RESIDUAL = 105`）：
```
gate_out, up_out → SiLU(gate)×up → [Linear] → out+residual → final
```
完整 MLP 后半段在 shared memory 中连续计算，避免 3 次 HBM 写入。

**实现文件**：
- Hopper: [tasks/hopper/norm_linear_hopper.cuh](include/mirage/persistent_kernel/tasks/hopper/norm_linear_hopper.cuh)
- Blackwell: [tasks/blackwell/norm_sm100.cuh](include/mirage/persistent_kernel/tasks/blackwell/norm_sm100.cuh)

---

### 4.5 Split-K GEMM

当 batch size 很小（decode 阶段典型 batch=1~8）时，矩阵 M 维极小，标准 GEMM 的 SM 利用率极低。Split-K 将 K（reduction）维切分给多个 SM 并行计算，再归约合并。

Python 侧配置（[persistent_kernel.py:1712](python/mirage/mpk/persistent_kernel.py#L1712)）：

```python
def splitk_linear_layer(self, input, weight, output, grid_dim, block_dim):
    # grid.y 即 split_k factor（K 切分数）
    if self.target_cc >= 100:
        self.kn_graph.register_task(tb_graph, "splitk_linear_sm100")      # =251
    elif self.target_cc >= 90:
        self.kn_graph.register_task(tb_graph, "splitk_linear_swapAB_hopper")  # =163
```

Qwen3 中 Split-K 仅在 B200（target_cc==100）启用（[qwen3/builder.py:260](python/mirage/mpk/models/qwen3/builder.py#L260)）：

```python
use_splitk = (target_cc == 100)
if use_splitk:
    self.mpk.splitk_linear_layer(
        input=self.attn_out, weight=self.w, output=self.attn_proj_out,
        grid_dim=(self.hidden_size // 128, 128 * 128 // self.hidden_size, 1),
        block_dim=(256, 1, 1),
    )
```

---

### 4.6 分页注意力（Paged Attention）

MPK 实现了受 vLLM 启发的分页 KV cache，支持多请求并发，每请求的 KV 存储在不连续的物理页中。

核心数据结构（CSR 格式，[runtime_header.h:342](include/mirage/persistent_kernel/runtime_header.h#L342)）：

```
paged_kv_indptr_buffer[i]     请求 i 的第一个 KV 页在 indices 中的起始偏移
paged_kv_indices_buffer[j]    第 j 个逻辑页映射到的物理页 ID
paged_kv_last_page_len[i]     请求 i 最后一页的有效 token 数
```

Python 侧按架构选择任务（[persistent_kernel.py:791](python/mirage/mpk/persistent_kernel.py#L791)）：

```python
if self.target_cc >= 100:
    task = "paged_attention_sm100"    # TASK_ATTN_SM100=257
elif self.target_cc >= 90:
    task = "paged_attention_hopper"   # TASK_PAGED_ATTENTION_HOPPER=153
else:
    task = "paged_attention"          # TASK_PAGED_ATTENTION_1/2=116/117
```

B200 上启用 Split-KV 两阶段执行：`TASK_PAGED_ATTENTION_SPLIT_KV_SM100=263` 计算分块注意力，`TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100=264` 合并 softmax 归一化结果。

---

### 4.7 MLA 注意力（DeepSeek V3 专用）

MLA（Multi-head Latent Attention）通过低秩压缩大幅减少 KV cache 内存占用。

关键常量（[tasks/blackwell/mla_decode_sm100.cuh:31](include/mirage/persistent_kernel/tasks/blackwell/mla_decode_sm100.cuh#L31)）：

```cpp
static constexpr int MLA_NUM_HEADS = 128;  // 注意力头数
static constexpr int MLA_D_K = 576;        // 压缩 KV 潜变量维度（vs GQA 的 128）
static constexpr int MLA_D_V = 512;        // 输出值维度
static constexpr int MLA_TILE_S = 128;     // 序列长度 tile
static constexpr int MLA_K_ITERS = 9;      // K 迭代次数（576/64=9）
static constexpr int MLA_V_CHUNKS = 8;     // V 块数（512/64=8）
```

两阶段执行：
1. **`TASK_MLA_DECODE_SM100=266`**：Q·K^T（576 维内积）→ softmax → 部分注意力权重
2. **`TASK_MLA_REDUCE_SM100=267`**：加权求和归约 + 输出投影，合并各 split 的部分结果

还支持张量并行变体：`TASK_MLA_MTP_DECODE_TP2/4/8_SM100`（[runtime_header.h:184](include/mirage/persistent_kernel/runtime_header.h#L184)）

---

### 4.8 FP8 量化

Blackwell 上的 E4M3 块量化，每 128 个元素共享一个 float32 scale factor：

```
BF16 激活 → [TASK_QUANTIZE_FP8_SM100=275] → FP8 激活 + scale
FP8 激活 + FP8 权重 → [TASK_LINEAR_FP8_SM100=276] → BF16 输出
```

实现文件：[tasks/blackwell/linear_fp8_sm100.cuh](include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100.cuh)（含 TMA 加速的 FP8 GEMM 调度器）

---

### 4.9 MoE（Mixture of Experts）

五个串联任务实现完整 MoE 前向：

```
输入 → [TASK_MOE_TOPK_SOFTMAX_SM100=260] 路由权重
     → [TASK_MOE_W13_LINEAR_SM100=254]  gate_out + up_out（W1·x 和 W3·x）
     → [TASK_SILU_MUL] SiLU(gate) × up
     → [TASK_MOE_W2_LINEAR_SM100=255]  各 expert 输出
     → [TASK_MOE_MUL_SUM_ADD_SM100=261] 加权求和 + 残差
```

`FullTaskDesc.task_metadata.expert_offset` 字段指定本任务负责的 expert 块偏移，多个 expert 的 GEMM 可并行分发到不同 SM。

---

### 4.10 投机解码（MTP Speculative Decoding）

```
TASK_MTP_PREPARE_VERIFY=274        准备验证上下文
TASK_MLA_MTP_DECODE_SM100=269      草稿模型 MLA 解码
TASK_MLA_MTP_REDUCE_SM100=270      草稿结果合并
TASK_MTP_VERIFY_STRICT=271         严格贪心验证
TASK_MTP_VERIFY_PROBABILISTIC=283  概率验证（温度>0）
TASK_MTP_ACCEPT_COMMIT=272         接受 token 并提交 KV cache
TASK_MTP_TOKEN_SCATTER=273         接受的 token 写回 cache
TASK_MTP_BUILD_EMBED_INPUT=294     为下一轮构建嵌入输入
```

---

### 4.11 多 GPU AllReduce

策略自动选择（[multigpu.py](python/mirage/mpk/multigpu.py)）：

```python
def auto_select_allreduce_implementation(target_cc, has_vmm, has_multicast):
    if target_cc >= 90 and has_vmm and has_multicast:
        return "nvshmem_tile"      # TASK_NVSHMEM_TILE_ALLREDUCE=302：NVLink Multicast 广播
    else:
        return "allgather_reduce"  # TASK_NVSHMEM_ALLGATHER_STRIDED_PUT=301：分片 put + 本地 reduce
```

Blackwell 特殊处理：为避免 `rdc=true`（relocatable device code）导致寄存器从 166 膨胀到 255，SM100 使用独立的设备端 allreduce 实现而非标准 NVSHMEM 设备库（[runtime_header.h:21-40](include/mirage/persistent_kernel/runtime_header.h#L21)）。

---

## 5. 模型支持：Qwen3 实现详解

### 5.1 模型注册机制

MPK 使用装饰器驱动的注册系统将模型名称映射到构建器类（[python/mirage/mpk/model_registry.py](python/mirage/mpk/model_registry.py)）：

```python
_MODEL_BUILDERS: dict = {}

def register_model_builder(*names):
    """支持多别名注册，一个构建器类可绑定多个模型名称"""
    def decorator(cls):
        for name in names:
            _MODEL_BUILDERS[name] = cls
        return cls
    return decorator

def get_builder(name: str):
    return _MODEL_BUILDERS[name]
```

**Qwen3 注册**（[qwen3/builder.py:14](python/mirage/mpk/models/qwen3/builder.py#L14)）：

```python
@register_model_builder(
    "Qwen3",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-0.8B"
)
class Qwen3Builder(GraphBuilder):
    ...
```

`MPK.build()` 通过 `get_builder(self.model_name)` 查找并实例化对应构建器。

---

### 5.2 GraphBuilder 抽象基类与 MirageModelConfig

**GraphBuilder**（[python/mirage/mpk/models/graph_builder.py:44](python/mirage/mpk/models/graph_builder.py#L44)）：

```python
class GraphBuilder(abc.ABC):
    def __init__(self, mpk: PersistentKernel, weights=None):
        self.mpk = mpk
        self.weights = weights or {}

    @abc.abstractmethod
    def build_from_model(self, model_path: str | None = None):
        raise NotImplementedError
```

所有模型构建器必须继承此类并实现 `build_from_model()`。`build_from_config()` 作为可选路径使用 `MirageModelConfig`。

**MirageModelConfig**（[graph_builder.py:8](python/mirage/mpk/models/graph_builder.py#L8)）是一个纯数据类，解耦了模型参数与 HuggingFace 模型对象：

```python
@dataclass
class MirageModelConfig:
    hidden_size: int = None
    intermediate_size: int = None
    vocab_size: int = None
    local_num_q_heads: int = None    # 张量并行后本 GPU 分配的 Q 头数
    local_num_kv_heads: int = None   # 张量并行后本 GPU 分配的 KV 头数
    head_dim: int = None
    num_layers: int = None
    k_cache: list[torch.Tensor] = None   # [num_layers] 每层 K cache GPU 张量
    v_cache: list[torch.Tensor] = None   # [num_layers] 每层 V cache GPU 张量
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None  # (cos, sin)
    state_dict: dict | None = None
    with_lm_head: bool = True
```

---

### 5.3 两条初始化路径

**路径一：`build_from_model()`（在线加载 HuggingFace 模型）**

```python
def build_from_model(self, model_name, model_path=None):
    # 1. 加载 HuggingFace 模型（从本地路径或 hub）
    self.config = AutoConfig.from_pretrained(model_path)
    self.model = Qwen3ForCausalLM(self.config)
    load_model(self.model, f"{model_path}/model{self.rank}-mp{self.world_size}.safetensors")

    # 2. 提取 state_dict 到 GPU
    state_dict = {k: v.cuda().to(torch.bfloat16) for k, v in self.model.state_dict().items()}

    # 3. 初始化 KV cache（GPU 上的 paged 物理页池）
    self.k_cache = [
        torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        for _ in range(num_layers)
    ]

    # 4. 计算 RoPE 位置嵌入（cos/sin）
    self.position_embeddings = compute_rotary_embeddings(max_seq_length, head_dim)

    # 5. 调用 build_from_dict() 完成图构建
    self.build_from_dict(state_dict, with_lm_head=True)
```

**路径二：`build_from_config()`（从预处理好的配置构建）**

```python
def build_from_config(self, model_config: MirageModelConfig):
    # 直接从 MirageModelConfig 读取所有参数
    self.hidden_size = model_config.hidden_size
    self.k_cache = model_config.k_cache
    self.position_embeddings = model_config.position_embeddings
    # ...
    self.build_from_dict(model_config.state_dict, model_config.with_lm_head)
```

两条路径最终汇合到 `build_from_dict()`，这是真正构建计算图的核心方法。

---

### 5.4 完整层序列：build_from_dict()

`build_from_dict()` 构建完整的 Qwen3 推理计算图（[qwen3/builder.py:524](python/mirage/mpk/models/qwen3/builder.py#L524)）。完整层序列如下：

```
① attach_input(input_tokens)         绑定输入 token ID 张量
② attach_input(cos/sin embeddings)   绑定 RoPE 位置编码
③ new_intermediate_tensors()         分配所有中间张量（GPU 静态分配）
④ attach_input(embed_tokens.weight)  绑定嵌入矩阵
⑤ embed_layer()                      Token → 嵌入向量 [T, hidden]

⑥ --- 循环 num_layers 次（build_layers）---

  [Attention Block]
  ⑥.1  attach_input(input_layernorm.weight)
  ⑥.2  rmsnorm_layer(x → rmsnorm_out)
  ⑥.3  linear_layer(rmsnorm_out, W_qkv → attn_in)   QKV 投影
  ⑥.4  paged_attention_layer(attn_in, k/v_cache → attn_out)
  ⑥.5a [B200] splitk_linear_layer(attn_out, W_o → x)  输出投影+残差
  ⑥.5b [H100] linear_with_residual_layer(attn_out, W_o, x → attn_proj_out)
  ⑥.6  [多GPU] allreduce_layer(attn_proj_out → attn_allreduce_out)

  [MLP Block]
  ⑥.7  attach_input(post_attention_layernorm.weight)
  ⑥.8  rmsnorm_layer(x → rmsnorm_out)
  ⑥.9  linear_layer(rmsnorm_out, W_gate_up → mlp_mid)  gate+up 合并投影
  ⑥.10 silu_mul_layer(mlp_mid → silu_mul_out)           SiLU(gate) × up
  ⑥.11a [B200] splitk_linear_layer(silu_mul_out, W_down → x)
  ⑥.11b [H100] linear_with_residual_layer(silu_mul_out, W_down, x → mlp_out)
  ⑥.12 [多GPU] allreduce_layer(mlp_out → mlp_final)

⑦ rmsnorm_layer(x, model.norm → rmsnorm_out)   最终归一化
⑧ linear_layer(rmsnorm_out, lm_head → argmax_in)  语言模型头
⑨ argmax_partial_layer(argmax_in → part_val, part_idx)  并行 argmax
⑩ argmax_reduce_layer(part_val, part_idx → output_tokens)  合并最终 token
```

---

### 5.5 QKV 权重 Shuffle 机制

Qwen3 使用 GQA（Grouped Query Attention），即多个 Q head 共享同一组 KV head。MPK 的 paged_attention_layer 要求权重按 KV head 分组交错排列，而 HuggingFace 原始权重是 [Q_all, K_all, V_all] 拼接。因此需要 shuffle。

**原始布局**（HuggingFace）：
```
W_qkv = [Q_head_0, Q_head_1, ..., Q_head_n,   # num_q_heads 行
          K_head_0, K_head_1, ..., K_head_m,   # num_kv_heads 行
          V_head_0, V_head_1, ..., V_head_m]   # num_kv_heads 行
```

**目标布局**（MPK 要求）：
```
W_qkv_shuffled = [Q_group_0, K_head_0, V_head_0,   # 第 0 个 KV group
                  Q_group_1, K_head_1, V_head_1,   # 第 1 个 KV group
                  ...]
# 其中 Q_group_i = (num_q_heads/num_kv_heads) 个连续 Q head
```

**三种 Shuffle 路径**（[qwen3/builder.py:270-318](python/mirage/mpk/models/qwen3/builder.py#L270)）：

**路径 A：已有 `qkv_proj.weight` + 分散 `q/k/v_proj.weight`（`inplace_shuffle_tensors`）**
```python
# 权重已拼接在同一张量，直接原地重排（CPU 侧完成，无额外内存分配）
inplace_shuffle_tensors(
    [q_proj.weight, k_proj.weight, v_proj.weight],  # 源视图（仅提供形状信息）
    qkv_proj.weight,                                 # 目标张量（原地重排）
    num_groups=self.num_local_kv_heads,              # KV group 数
    shuffled_dim=0,
)
```

**路径 B：`online_notoken` 模式（`shuffle_tensors` CPU 合并新张量）**
```python
# 分散的 q/k/v 权重，先在 CPU 合并+shuffle 为新张量，再 attach
self.w_qkv_tensor = shuffle_tensors(
    [q_proj.weight, k_proj.weight, v_proj.weight],
    num_groups=self.num_local_kv_heads,
    shuffled_dim=0,
)
self.shuffled_tensors[f"layer_{i}_qkv_proj"] = self.w_qkv_tensor  # 防止 GC 回收
```

**路径 C：其他模式（`mpk.shuffle_tensors` 在任务图中插入 shuffle 任务）**
```python
# 在计算图中插入显式 shuffle 节点，推理时由 GPU 执行
w_qkv = self.mpk.shuffle_tensors(
    inputs=[w_q, w_k, w_v],
    shuffled_dim=0,
    num_groups=self.num_local_kv_heads,
    name=f"layer_{i}_qkv_proj",
)
```

gate+up 投影的 shuffle 逻辑完全对称，区别是 `num_groups=rmsnorm_num_tasks//2`（按 GEMM tile 数分组，而非 KV head 数）。

---

### 5.6 中间张量静态分配

`new_intermediate_tensors()` 在 `build_from_dict()` 开始时一次性分配所有中间张量（[qwen3/builder.py:114](python/mirage/mpk/models/qwen3/builder.py#L114)）。这是 MPK 消除推理时动态内存分配的关键：

```python
def new_intermediate_tensors(self):
    T = self.mpk.max_num_batched_tokens  # 最大 token 数（编译时常量）
    
    # 嵌入输出：[T, hidden_size]
    self.y = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
    
    # RMSNorm 输出（各层复用同一块内存）
    self.rmsnorm_out = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
    
    # Attention 输入（QKV 投影输出）：[T, (nq+2*nkv)*head_dim]
    self.attn_in = self.mpk.new_tensor(dims=(T, self.fused_outdim_1), dtype=bfloat16)
    
    # Attention 输出：[T, nq*head_dim]
    self.attn_out = self.mpk.new_tensor(dims=(T, self.num_local_q_heads * self.head_dim), dtype=bfloat16)
    
    # MLP 中间：gate+up 合并输出 [T, 2*intermediate_size]
    self.mlp_mid = self.mpk.new_tensor(dims=(T, self.fused_outdim_2), dtype=bfloat16)
    
    # SiLU×MUL 输出：[T, intermediate_size]
    self.silu_mul_out = self.mpk.new_tensor(dims=(T, self.intermediate_size), dtype=bfloat16)
    
    # argmax 中间结果
    self.argmax_in = self.mpk.new_tensor(dims=(T, self.padded_vocab_size), dtype=bfloat16)
    self.argmax_part_value = self.mpk.new_tensor(...)
    self.argmax_part_index = self.mpk.new_tensor(...)
    
    # 多 GPU AllReduce 缓冲（仅 world_size > 1 时分配）
    if self.world_size > 1:
        self.allreduce_buf = self.mpk.new_tensor(...)
```

所有层**复用**同一批中间张量（流水线执行保证不冲突），整个模型推理期间不发生任何动态内存分配。

---

### 5.7 与 DeepSeek V3 的关键差异

DeepSeek V3（[python/mirage/mpk/models/deepseek_v3/builder.py](python/mirage/mpk/models/deepseek_v3/builder.py)）相比 Qwen3 的主要区别：

| 特性 | Qwen3 | DeepSeek V3 |
|------|-------|-------------|
| 注意力机制 | GQA（标准 Q/K/V 投影） | MLA（低秩压缩 KV，D_K=576） |
| 权重精度 | BF16 | FP8（E4M3，per-block scale） |
| MLP 类型 | Dense FFN（gate+up+down） | MoE（topk routing + 专家 GEMM） |
| KV cache 维度 | head_dim × num_kv_heads | MLA_D_K=576（压缩潜变量） |
| 投机解码 | 可选 argmax MTP | 完整 MTP 流水线（4 专用任务） |
| 任务类型 | TASK_ATTN_SM100 | TASK_MLA_DECODE/REDUCE_SM100 |

---

## 6. 性能优化机制

### 6.1 SM 分区与任务并发

MPK 的性能基础是精确的 SM 分区，使 Worker SM 和 Scheduler SM 同时满载（[python/mirage/utils.py:33](python/mirage/utils.py#L33)）：

| GPU | 总 SM | Worker SM | Scheduler SM |
|-----|-------|-----------|--------------|
| B200 | 148 | 144 | 16 |
| H100 | 132 | 128 | 16 |
| A100 | 108 | 96 | 48 |
| A6000 | 84 | 64 | 80 |

Scheduler:Worker 约 1:8 比例，保证调度带宽不成为瓶颈。`scheduler = 4 * (sm_cnt - worker)` 即 scheduler 数为剩余 SM 的 4 倍（每个 SM 跑 4 个 warp 组）。

### 6.2 编译缓存（Compilation Cache）

MPK 编译通常需要 3~10 分钟（nvcc 编译大量模板实例化），因此实现了精确校验的编译缓存（[persistent_kernel.py:474](python/mirage/mpk/persistent_kernel.py#L474)）。

保存的元数据字段（`kernel_metadata.json`）：

```json
{
  "mode": "offline",
  "max_seq_length": 4096,
  "max_num_batched_requests": 32,
  "max_num_batched_tokens": 32,
  "max_num_pages": 1024,
  "page_size": 16,
  "world_size": 1,
  "rank": 0,
  "cuda_cc": 100,
  "tensor_names": ["embed_tokens", "layer_0_qkv_proj", ...]
}
```

加载时 `_validate_kernel_compatibility()` 逐一对比所有字段，任一不匹配则触发重新编译。这是必要的——因为批处理参数全部是编译时宏定义，会直接影响 shared memory 分配大小和循环展开。

### 6.3 Shared Memory 预算管理

不同架构和模式下的动态 shared memory 上限（[runtime_header.h:46-83](include/mirage/persistent_kernel/runtime_header.h#L46)）：

| 架构 | 模式 | 动态 Smem 上限 |
|------|------|----------------|
| SM90/SM100 | offline | 207 KB - 静态预留 |
| SM90/SM100 | online_notoken | 220 KB - 静态预留 |
| SM80 (A100) | offline/online | 163 KB - 静态预留 |

Hopper/Blackwell 额外预留 6 KB 静态 shared memory（TMA 描述符等），Ampere 只需 3 KB。

### 6.4 三种服务模式对比

| 模式 | 编译宏 | 特点 |
|------|--------|------|
| `offline` | `MODE_OFFLINE` | 所有请求预填充后批量解码，适合离线 benchmark |
| `online` | `MODE_ONLINE` | CPU 传 request_id 到 GPU 队列，标准在线服务 |
| `online_pinned` | `MODE_ONLINE_PINNED` | 锁自由 pinned memory 环形缓冲，CPU-GPU 零拷贝通信 |

`online_pinned` 模式下 CPU 写入 `pinned_inbox_tokens` 后仅设置 `pinned_req_ready=1`（release 语义），GPU 通过 acquire 语义读取，无须系统调用或 `cudaMemcpy`，端到端延迟最低。

### 6.5 多级流水线与延迟隐藏

MPK 在多个层次同时进行流水线：

1. **TMA 流水线（Kstages）**：TMA 异步加载 K+1 轮数据时，WGMMA 同步计算第 K 轮
2. **任务间流水线**：Scheduler 在 Worker 执行当前任务时预填充下一批任务到 worker_queue
3. **MLA Split-KV 并行**：多个 `TASK_MLA_DECODE_SM100` 并行处理不同 KV 分块，merge 任务等所有 decode 完成后执行
4. **AllReduce 重叠**：通过精确的 `dependent_event` 控制，AllReduce 的 NVSHMEM put 可与下一层 RMSNorm 重叠

---

## 7. 新模型接入完整指南

### 7.1 接入前提确认

开始前确认模型所需算子是否已在 `TaskType` 枚举中存在：

- 标准 GQA 注意力 → `TASK_ATTN_SM100` / `TASK_PAGED_ATTENTION_HOPPER`
- MLA 注意力 → `TASK_MLA_DECODE_SM100` + `TASK_MLA_REDUCE_SM100`
- Dense FFN → `TASK_LINEAR_SM100` + `TASK_SILU_MUL`
- MoE FFN → `TASK_MOE_TOPK_SOFTMAX_SM100` 等 5 个任务
- FP8 量化 → `TASK_QUANTIZE_FP8_SM100` + `TASK_LINEAR_FP8_SM100`

若需要全新算子，需先在 `include/mirage/persistent_kernel/tasks/{arch}/` 下实现 CUDA 内核，并在 `runtime_header.h` 注册新 `TaskType`，再在 `persistent_kernel.cuh` 的 `_execute_task()` dispatch 表中添加 case。

### 7.2 第一步：创建 builder 文件并注册

```
python/mirage/mpk/models/
└── my_model/
    ├── __init__.py
    └── builder.py
```

```python
# python/mirage/mpk/models/my_model/builder.py
from ..graph_builder import GraphBuilder, MirageModelConfig
from ...model_registry import register_model_builder

@register_model_builder("MyModel", "org/my-model-7B")
class MyModelBuilder(GraphBuilder):
    def __init__(self, mpk, weights=None):
        super().__init__(mpk, weights)
        # 架构参数（build_from_model 或 build_from_config 时填充）
        self.hidden_size = None
        self.intermediate_size = None
        self.num_layers = None
        self.num_local_q_heads = None
        self.num_local_kv_heads = None
        self.head_dim = None
        self.vocab_size = None
```

在 `python/mirage/mpk/models/__init__.py` 中导入：

```python
from .my_model import builder as my_model_builder  # 触发注册装饰器
```

### 7.3 第二步：实现两条初始化路径

```python
def build_from_model(self, model_name: str, model_path: str | None = None):
    # 1. 加载 HuggingFace 模型
    config = AutoConfig.from_pretrained(model_path or model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path or model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    state_dict = dict(model.state_dict())

    # 2. 填充架构参数
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.num_layers = config.num_hidden_layers
    self.num_local_q_heads = config.num_attention_heads // self.mpk.world_size
    self.num_local_kv_heads = config.num_key_value_heads // self.mpk.world_size
    self.head_dim = self.hidden_size // config.num_attention_heads
    self.vocab_size = config.vocab_size

    # 3. 分配 KV cache（形状需与 paged_attention_layer 一致）
    self.k_cache = [
        torch.zeros(self.mpk.max_num_pages, self.mpk.page_size,
                    self.num_local_kv_heads, self.head_dim,
                    dtype=torch.bfloat16, device="cuda")
        for _ in range(self.num_layers)
    ]
    self.v_cache = [k.clone() for k in self.k_cache]

    # 4. 计算 RoPE 嵌入
    self.position_embeddings = self._compute_rotary_embeddings()

    # 5. 构建计算图
    self.build_from_dict(state_dict, with_lm_head=True)

def build_from_config(self, model_config: MirageModelConfig):
    self.hidden_size = model_config.hidden_size
    self.intermediate_size = model_config.intermediate_size
    self.num_layers = model_config.num_layers
    self.num_local_q_heads = model_config.local_num_q_heads
    self.num_local_kv_heads = model_config.local_num_kv_heads
    self.head_dim = model_config.head_dim
    self.k_cache = model_config.k_cache
    self.v_cache = model_config.v_cache
    self.position_embeddings = model_config.position_embeddings
    self.build_from_dict(model_config.state_dict, model_config.with_lm_head)
```

### 7.4 第三步：分配中间张量

```python
def new_intermediate_tensors(self):
    from ....core import bfloat16
    T = self.mpk.max_num_batched_tokens

    self.rmsnorm_out   = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
    self.attn_in       = self.mpk.new_tensor(
        dims=(T, (self.num_local_q_heads + 2*self.num_local_kv_heads) * self.head_dim),
        dtype=bfloat16)
    self.attn_out      = self.mpk.new_tensor(
        dims=(T, self.num_local_q_heads * self.head_dim), dtype=bfloat16)
    self.attn_proj_out = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
    self.mlp_mid       = self.mpk.new_tensor(dims=(T, 2 * self.intermediate_size), dtype=bfloat16)
    self.silu_mul_out  = self.mpk.new_tensor(dims=(T, self.intermediate_size), dtype=bfloat16)
    self.mlp_out       = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
    self.argmax_in     = self.mpk.new_tensor(dims=(T, self.vocab_size), dtype=bfloat16)
    self.argmax_part_value = self.mpk.new_tensor(dims=(self.mpk.num_workers, T), dtype=bfloat16)
    self.argmax_part_index = self.mpk.new_tensor(dims=(self.mpk.num_workers, T), dtype=bfloat16)

    if self.mpk.world_size > 1:
        self.allreduce_buf      = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
        self.attn_allreduce_out = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
        self.mlp_final          = self.mpk.new_tensor(dims=(T, self.hidden_size), dtype=bfloat16)
```

### 7.5 第四步：实现 build_from_dict()

```python
def build_from_dict(self, state_dict: dict, with_lm_head: bool):
    target_cc = (torch.cuda.get_device_properties(0).major * 10
                 + torch.cuda.get_device_properties(0).minor)
    use_splitk = (target_cc == 100)

    # 绑定输入/输出张量
    self.x = self.mpk.attach_input(self.mpk.meta_tensors["input_tokens"], "input_token")
    self.cos = self.mpk.attach_input(
        self.position_embeddings[0][0, :self.mpk.max_seq_length, :], "cos_position_embedding")
    self.sin = self.mpk.attach_input(
        self.position_embeddings[1][0, :self.mpk.max_seq_length, :], "sin_position_embedding")
    argmax_out = self.mpk.attach_input(self.mpk.meta_tensors["output_tokens"], "output_token")

    self.new_intermediate_tensors()
    self.y = self.mpk.new_tensor(dims=(self.mpk.max_num_batched_tokens, self.hidden_size))

    # Embedding 层
    w_embed = self.mpk.attach_input(state_dict["model.embed_tokens.weight"], "embed_tokens")
    self.mpk.embed_layer(input=self.x, weight=w_embed, output=self.y,
                         grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)
    self.x = self.y

    # Transformer 层循环
    for i in range(self.num_layers):
        self._build_one_layer(state_dict, i, use_splitk)

    # 最终 RMSNorm + LM Head + Argmax
    if with_lm_head:
        w_norm = self.mpk.attach_input(state_dict["model.norm.weight"], "model_norm_weight")
        self.mpk.rmsnorm_layer(input=self.x, weight=w_norm, output=self.rmsnorm_out,
                               grid_dim=(self.mpk.max_num_batched_tokens, 1, 1),
                               block_dim=(128, 1, 1))
        w_lm = self.mpk.attach_input(state_dict["lm_head.weight"], "lm_head")
        self.mpk.linear_layer(input=self.rmsnorm_out, weight=w_lm, output=self.argmax_in,
                              grid_dim=(self.vocab_size // 64, 1, 1), block_dim=(128, 1, 1))
        self.mpk.argmax_partial_layer(
            input=self.argmax_in, output=(self.argmax_part_value, self.argmax_part_index),
            grid_dim=(self.mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
        self.mpk.argmax_reduce_layer(
            input=(self.argmax_part_value, self.argmax_part_index), output=argmax_out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
```

### 7.6 第五步：单层构建与 QKV shuffle

每层的结构（Attention + MLP）封装在 `_build_one_layer()` 中，QKV shuffle 逻辑参考 [5.5 节](#55-qkv-权重-shuffle-机制)。关键点：

- **GQA 模型**（Qwen3/Llama3）：`num_groups = num_local_kv_heads`
- **已融合 qkv_proj**（如从 SGLang/vLLM 导出）：用 `inplace_shuffle_tensors`
- **split q/k/v 权重**：默认模式用 `mpk.shuffle_tensors`（图内 GPU shuffle），`online_notoken` 模式用 CPU 端 `shuffle_tensors` 预处理

### 7.7 第六步：验证

```python
from mirage.mpk import MPK, MPKMetadata

meta = MPKMetadata(
    model_name="MyModel",
    mode="offline",
    max_seq_length=512,
    max_num_batched_requests=1,
    max_num_batched_tokens=1,
    max_num_pages=64,
    page_size=16,
)
mpk = MPK(meta)
mpk.build()    # 构建计算图
mpk.compile()  # nvcc 编译（首次 3~10 分钟，后续命中缓存秒级）
output = mpk(input_tokens=..., ...)
```

### 7.8 支持全新算子的扩展步骤

若模型需要当前不支持的算子：

1. **在 `runtime_header.h` 添加 TaskType**（选择合适架构区段）
2. **在 `tasks/{arch}/my_op.cuh` 实现 CUDA 内核**（参考 `linear_sm100_mpk.cuh` 模板）
3. **在 `persistent_kernel.cuh` 的 `_execute_task()` 添加 dispatch case**：
   ```cpp
   case TASK_MY_NEW_OP_SM100:
       kernel::my_new_op_sm100_impl(task.input_ptrs, task.output_ptrs);
       break;
   ```
4. **在 `persistent_kernel.py` 添加 layer 方法**：构建 TBGraph，调用 `register_task()`
5. 重新编译内核（`mpk.compile()`），在 builder 中调用新 layer 方法

---

## 总结

MPK 通过以下核心技术实现高效 LLM 推理：

1. **Megakernel**：一次 kernel launch 完成整个推理迭代，消除频繁 CPU-GPU 同步开销
2. **事件驱动调度**：Worker/Scheduler SM 分工，原子事件计数器实现无锁依赖追踪
3. **架构分层任务**：Ampere/Hopper/Blackwell 各有专属实现，充分利用 TMA、WGMMA、UMMA、TMEM 等硬件特性
4. **算子融合**：RMSNorm+Linear、SiLU×MUL+Linear+Residual 等融合消除中间结果 HBM 写入
5. **Paged KV Cache**：GPU 端 CSR 格式管理，Scheduler SM 内部分配物理页，无 CPU-GPU 同步
6. **静态内存分配**：所有中间张量在构建阶段一次性分配，推理期间零动态分配

接入新模型的核心步骤是：注册构建器 → 分配中间张量 → 按层调用 layer 方法 → 处理权重 shuffle → 编译验证。整个框架的扩展点非常清晰，新模型接入主要是 Python 层的组合工作，只有在需要全新算子时才需要下沉到 CUDA 内核实现。

