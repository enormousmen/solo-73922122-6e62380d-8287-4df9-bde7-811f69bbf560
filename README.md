# Resumable Training API（可恢复单作业训练服务）

FastAPI + CPU 版 PyTorch 实现的**单作业训练服务**。训练工程师在作业被抢占（暂停、
进程强退、容器被杀）后，可以从最近成功的检查点继续训练；在**同环境、同种子、同
配置**下，恢复后的逐批损失与最终参数与**不中断训练逐位一致**。

- 模型：两层全连接网络 `Linear → ReLU → Dropout → Linear`
- 优化器：Adam；每轮（epoch）打乱全部样本后按批训练
- 数据：仓库内置固定分类数据集 `data/classification.csv`（600 样本 / 2 特征 / 3 类）
- 默认配置：`config/default.json`（可在启动时按字段覆盖）

## 目录结构

```
app/
  main.py        # FastAPI 入口（/docs 自动生成）
  trainer.py     # 单作业状态机 + 确定性训练循环 + 暂停/续训 + 漂移基线生命周期
  checkpoint.py  # 原子检查点读写（tmp + fsync + rename）
  drift.py       # 五分位桶 + PSI 计算 + 不可变漂移基线（随检查点持久化）
  config.py      # 默认配置加载与校验
  dataset.py     # 固定 CSV 数据集加载
  model.py       # 两层 FC + Dropout
config/default.json      # 默认训练配置
data/classification.csv  # 固定分类数据集（scripts/generate_data.py 生成）
scripts/generate_data.py # 数据集生成脚本（固定种子，仅作溯源）
scripts/verify_repro.py  # 端到端验证：不中断对照 / 暂停续训 / 强退恢复 等
scripts/verify_drift.py  # 漂移接口端到端验证：PSI 数学 / 基线持久化 / 换作业 / 不阻塞
Dockerfile, docker-compose.yml, requirements.txt
```

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务: http://localhost:8000  交互式文档: http://localhost:8000/docs
```

检查点保存在命名卷 `trainer-runs` 中，容器被杀或重启后数据不丢。

### 本地运行（开发）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 环境变量: RUNS_DIR(默认 runs/)  CONFIG_PATH(默认 config/default.json)
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息 |
| GET | `/config` | 默认配置与运行时信息 |
| POST | `/train/start` | 启动新训练。Body: `{"force": false, "config": {覆盖字段}}` |
| POST | `/train/pause` | 暂停。**训练停止且完整检查点持久化后才返回 200** |
| POST | `/train/resume` | 从暂停态/最近检查点续训（并发只放行一个训练者） |
| GET | `/train/status` | 状态与进度（epoch/batch/百分比/检查点健康度） |
| GET | `/train/losses` | 逐批损失（支持 `since_step`、`limit`） |
| GET | `/train/parameters` | 下载当前模型参数（.pt，取自最近持久化检查点） |
| GET | `/train/checkpoint` | 下载完整检查点（模型+优化器+RNG+历史） |
| POST | `/train/drift` | 线上无标签批次的 PSI 分布漂移评分（20～200 行，按训练 CSV 列序） |

交互式文档（Swagger UI）在 `/docs`，可直接在浏览器里调通上述接口。

### curl 示例

```bash
# 启动（覆盖部分配置；已有运行需 force=true 或先续训）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8, "batch_size": 32, "lr": 0.01}}'

# 查询进度
curl -s localhost:8000/train/status | jq '.state, .completed_batches, .total_batches'

# 逐批损失
curl -s 'localhost:8000/train/losses?limit=5' | jq '.losses[]'

# 暂停（阻塞至训练停止且检查点落盘才返回）
curl -X POST localhost:8000/train/pause

# 续训
curl -X POST localhost:8000/train/resume

# 下载参数 / 完整检查点
curl -OJ localhost:8000/train/parameters     # model_parameters.pt
curl -OJ localhost:8000/train/checkpoint     # checkpoint.pt
```

### 分布漂移检测（POST /train/drift）

线上拿到无标签样本后，用 PSI（Population Stability Index）判断输入是否已
偏离该作业训练时的分布。请求体为 20～200 条**有限数值向量**，每条严格按训练
CSV 的特征列顺序排列（本数据集即 `[feature_1, feature_2]`）：

```bash
curl -X POST localhost:8000/train/drift -H 'Content-Type: application/json' \
     -d '{"samples": [[0.12, -1.03], [-0.77, 0.44], ... ]}'
```

计算口径（实现见 `app/drift.py`，逐列独立计算）：

1. **五分位桶**：以该作业实际加载的 CSV 为基线，取每列 20/40/60/80% 分位数
   作为内部分桶切点；相邻切点重合（离散/常量列）时合并，桶数可少于 5，常量
   列退化为 1 个兜底桶；两端桶开口，**超出训练观测范围的值自动归入两端桶**。
2. **平滑归一化**：基线与新批次的各桶计数各自加 `1e-6` 后重新归一化（和为
   1），空桶因此也得到非零占比，对数始终有定义。
3. **PSI**：`sum_b (p_new - p_base) * ln(p_new / p_base)`，逐列返回分数；
   **严格大于 0.2** 的列进入 `alerts`，同时返回每列切点、两侧计数/占比、
   超范围样本数，以及基线 `run_id`、保存时的 epoch/批次（`saved_*`）。

基线生命周期与不可变性：

- 基线在作业启动时从**当前实际加载的 CSV** 构建，但只处于 pending 状态；
  它随**第一个成功落盘的检查点**一起持久化后才对外发布（
  `GET /train/status` 的 `drift_baseline_available` 可见）。
- 基线对象只含普通 Python 类型，整体存进检查点；重启后**只从检查点还原，
  绝不根据磁盘上的 CSV 重算**——即使 CSV 已被改写。响应中的
  `dataset_signature_matches_current`（CSV 的 SHA-256 比对）会显式提示文件
  是否已被改写，但历史基线与评分保持不变。
- 无检查点（从未训练）或 `force` 新作业在首个检查点落盘前调用，返回
  **404** 与结构化原因 `{"error": "drift_baseline_unavailable", ...}`，不
  返回未持久化的基线。
- 请求读取基线用的是不可变快照；即使随后 `force` 换了作业，已在处理中的
  请求仍按**旧 `run_id`** 完成计算，并以 `baseline_stale=true` 与
  `stale_reason` 标注，不会抛错或阻塞。
- 评分在 FastAPI 工作线程中进行且只短暂持有状态锁，训练线程与
  `/train/status` 等查询不会被漂移计算阻塞。

```bash
# 无可用基线时
curl -s -X POST localhost:8000/train/drift -H 'Content-Type: application/json' \
     -d '{"samples": [[0,0],[0,0]]}' | head
# 404 -> {"detail": {"error": "drift_baseline_unavailable",
#          "message": "no drift baseline available: ... first successful checkpoint"}}
```

自动化验证：`python3 scripts/verify_drift.py`（47 项检查，覆盖 PSI 数学与
独立参考实现交叉校验、入参校验、404 语义、杀进程重启 + CSV 改写不重算、
换作业旧 `run_id` 完成并标注 stale、高并发下不阻塞训练与状态查询、`/docs`）。

状态机：`idle → running ⇄ pausing → paused → running … → finished`（异常为
`failed`）。`paused` 表示训练已停、可续训；`checkpoint_dirty=true` 表示内存进度
领先于磁盘（上次保存失败），此时重启进程会回退到磁盘上的旧检查点。

## 可复现性设计（为什么恢复后逐位一致）

1. **单线程 + 确定性算子**：`torch.set_num_threads(1)` 并开启
   `torch.use_deterministic_algorithms(True)`，同一构建下浮点结果逐位确定。
2. **单一 RNG 流**：新运行先 `torch.manual_seed(seed)`，模型初始化与 Dropout
   都从该流取数；检查点保存 `torch.get_rng_state()`，恢复时原样写回。
3. **每轮打乱可重推**：第 e 轮的样本排列是 `(seed, e)` 的纯函数
   （`random.Random(f"{seed}:{epoch}")`），重启后无需存储即可重建当轮顺序。
4. **批边界检查点**：每个检查点记录 `epoch`、`next_batch`、模型与 Adam 状态、
   RNG 状态、完整损失历史、解析后的配置与 `completed` 标志——即"准备执行第 k
   批"的完整现场。恢复 = 还原现场后继续同一确定性序列。
5. **原子写**：先写 `.checkpoint.pt.tmp.<uuid>` → fsync → `os.replace` 原子改名
   → 目录 fsync。保存中被强退，磁盘上只会是**完整的旧点或完整的新点**；
   启动时清理残留的临时文件。保存失败不触碰旧文件，明确报错，可重试。

> 注意：逐位一致的前提是**同环境**（同一 torch 构建/CPU 指令集、单线程）。跨
> 不同 CPU 架构或 torch 版本不保证逐位一致，但同机同镜像（如本 compose）成立。

## 不中断对照与退出重启示例

### 自动化端到端验证（推荐）

```bash
python3 scripts/verify_repro.py
```

脚本会真实拉起 uvicorn 服务进程，依次验证（详见脚本 docstring）：
A 不中断基线；B 暂停+续训；C 强退+重启+续训；D 并发续训只放行一个；
E1 保存失败→报错→旧点保留→重试成功；E2 保存失败后强退→回退旧点恢复；
F 慢保存不挡查询；G 保存进行中强退→恢复完整旧/新点；H 基础接口与 /docs。
B–E、G 的逐批损失与最终参数均与各自不中断基线**逐位一致**才判通过。

> 本环境的 seccomp 策略禁止向其他进程发信号，脚本用 `ENABLE_TEST_CRASH_ENDPOINT=1`
> 开启的测试专用 `POST /test/crash`（`os._exit`，立即死亡、无任何清理）模拟强退；
> 恢复语义与 SIGKILL 等价。该端点默认关闭，切勿在生产开启。

### Docker 手动退出重启示例

```bash
docker compose up -d --build
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8}}'

docker kill resumable-trainer          # 模拟抢占/强退（真实 SIGKILL）
docker compose start                   # 重启容器（检查点在命名卷中）
curl -s localhost:8000/train/status    # state=paused, recovered_after_restart=true
curl -X POST localhost:8000/train/resume
curl -s localhost:8000/train/status    # 直至 state=finished
```

另开一个终端可对同一数据集做一次不中断对照（`force=true` 重跑同配置），用
`GET /train/losses` 与 `GET /train/parameters` 对比两边逐批损失与参数完全一致。

## 实测结果

### 可恢复性（`python3 scripts/verify_repro.py`）

（以下为本仓库环境下的真实输出摘要）

```
    workdir: /tmp/verify-repro-ndor_el1
    
    == H: service basics (idle server) ==
      [PASS] H1 GET /
      [PASS] H2 GET /docs (Swagger UI)
      [PASS] H3 parameters before any checkpoint -> 404
      [PASS] H4 resume with no run -> 404
      [PASS] H5 pause while idle -> 409
    
    == A: uninterrupted baseline ==
      [PASS] A baseline finished - 95 batches, final loss 0.027771
    
    == B: pause + resume ==
      [PASS] B1 pause returns only after checkpoint persisted - pause latency 0.03s, stopped at batch 14
      [PASS] B2 resume accepted
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] B3 pause+resume == uninterrupted
    
    == D: concurrent resume starts exactly one trainer ==
      [PASS] D1 concurrent resumes -> one 200, one 409 - got [200, 409]
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] D2 resumed run == uninterrupted
    
    == E1: checkpoint save failure keeps old checkpoint, retry works ==
      [PASS] E1 pause with failing save -> 500 - training is stopped but the checkpoint could not be persisted; the previous checkpoint is intact on 
      [PASS] E1 status shows dirty checkpoint + error
      [PASS] E1 old checkpoint untouched
      [PASS] E1 retried pause -> 200, checkpoint persisted
      [PASS] E1 resume after retry
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E1 run after save-failure == uninterrupted
    
    == F: slow checkpoint saves do not block queries ==
      [PASS] F1 queries respond fast while a 1s save is in flight - worst status latency 1ms
      [PASS] F2 pause during slow saves
      [PASS] F3 slow-save run finishes
    
    == C: hard crash + restart recovers from latest checkpoint ==
      [PASS] C1 server crashed mid-run
      [PASS] C2 restart recovers paused state from checkpoint - recovered at batch 10/95
      [PASS] C3 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] C4 kill+restart == uninterrupted
    
    == E2: save failure, then hard crash -> old checkpoint used ==
      [PASS] E2 pause with failing save -> 500
      [PASS] E2 restart falls back to old complete checkpoint - recovered at batch 6/95
      [PASS] E2 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E2 save-fail+kill == uninterrupted
    
    == G: hard crash in the middle of a checkpoint save ==
      (baseline: identical config, uninterrupted)
      [PASS] G1 server crashed during save
      [PASS] G2 no torn checkpoint / stale temp cleaned - leftovers: none
      [PASS] G3 restart recovers a complete checkpoint (old or new) - recovered at batch 1/19
      [PASS] G3b resume after restart
        losses: 19 entries, exact match
        params: all tensors bit-identical (sha256 1353a7415c7cae90 == 1353a7415c7cae90)
      [PASS] G4 kill-mid-save == uninterrupted
    
    ================ SUMMARY ================
      33/33 checks passed（逐项明细同上，全部 PASS）

```

### 漂移检测（`python3 scripts/verify_drift.py`）

```
    == M: quantile bucket + PSI math ==              13/13 PASS
      五分位切点 / 重合边界合并 / 常量列兜底桶 / 范围外归入两端桶 /
      1e-6 平滑归一化 / PSI 公式与独立参考实现逐位一致
    == S: 换作业后在途请求仍按旧 run_id 完成          3/3  PASS
      baseline_stale=true 显式标注；新作业存点前明确 404
    == U: 无可用基线                                  3/3  PASS
      从未训练 / force 新作业首存前 -> 404 结构化原因；存点后状态翻转
    == V: 请求校验                                    7/7  PASS
      20~200 行边界、NaN/Inf（422 且错误体 JSON 安全）、异形行、列宽不符
    == D: HTTP 漂移评分                               7/7  PASS
      同分布不告警 / 偏移双列告警 / 严格 >0.2 / 20 与 200 行边界
    == N: 不阻塞训练与状态查询                        3/3  PASS
      3 路并发漂移 + 训练中，status p95 约 10ms，训练完整跑完
    == N-lock: 锁内仅做基线快照                       1/1  PASS
      50 次漂移共 100 个临界区，单次持锁最大约 0.015ms，评分完全在锁外
    == H: /docs 与 OpenAPI                            2/2  PASS
    == S-HTTP: force 换作业的在途请求语义             3/3  PASS
      首存前 404 不复活旧基线；新作业发布后按新 run_id 服务且不标 stale
    == P: 基线随检查点持久化、抗 CSV 改写             5/5  PASS
      杀进程重启后 run_id 与切点不变；改写 CSV 不重算，仅签名标记不一致

    ================ SUMMARY ================
      47/47 checks passed
```

自动化验证：`python3 scripts/verify_drift.py`（47 项检查，覆盖 PSI 数学与
独立参考实现交叉校验、入参校验、404 语义、杀进程重启 + CSV 改写不重算、
换作业旧 `run_id` 完成并标注 stale、高并发下不阻塞训练与状态查询、`/docs`）。

## 限制与未完成事项

- 单作业、单进程、单 worker：多副本/多 worker 部署会破坏单训练者语义（需引入
  外部锁与共享存储，未实现）。
- 逐位一致仅保证同环境同构建（见上节）；未做跨机器/跨 torch 版本对齐。
- 检查点只保留最新一份（原子替换保证新旧二选一）；未做历史版本保留/滚动清理。
- `force=true` 启动新运行会**立即丢弃**旧检查点：若新运行在写出第一个检查点
  前崩溃，则无点可恢复（状态回到 `idle`），只能重新开始——这是刻意语义，
  避免"已丢弃的旧运行"在重启后被悄悄恢复。
- `GET /train/parameters` 取自最近**已持久化**检查点，运行中可能落后最多
  `checkpoint_every` 批；未提供"当前内存参数"下载（避免与训练线程竞争）。
- 漂移基线与检查点同生命周期：`force=true` 丢弃旧检查点的同时旧基线也一并
  失效，新作业在首个成功检查点落盘前 `/train/drift` 明确返回 404；这是与
  "旧运行不得被悄悄恢复"一致的刻意语义。
- 漂移分桶基于 numpy 线性分位数；离散列在切点重合时会产生空桶（由 `1e-6`
  平滑兜底），常量列只有一个兜底桶、PSI 恒为 0（无法识别常量列内部的漂移）。
- 本环境无 docker，镜像构建与 compose 未实际执行；Dockerfile/compose 与本地实
  测使用完全相同的启动命令与依赖版本，但未在真实 Docker 中跑过验证脚本。
- 暂停等待上限为 `pause_timeout_seconds`（默认 30s），超时返回 504，训练仍会
  在批边界停下，可轮询状态确认。
