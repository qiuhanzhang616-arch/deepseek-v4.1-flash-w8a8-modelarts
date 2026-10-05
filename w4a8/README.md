# 在公有云 ModelArts Standard 部署 DeepSeek-V4.1-Flash W4A8（A2）

> English runbook for the currently verified four-replica production profile:
> [DeepSeek-V4.1-Flash W4A8 Production Deployment Guide](PRODUCTION_DEPLOYMENT_EN.md).

本指南面向第一次接手部署的工程师，覆盖权重和镜像来源、脚本打包、SWR 推送、SFS Turbo、ModelArts 实时服务、验收与回滚。它记录的是 **Ascend A2 上验证过的一种 W4A8 单机副本方案**，不是适用于所有区域、驱动和镜像版本的通用参数。请先在隔离服务验证，勿直接覆盖生产路由。

> 权重是社区发布方 [chiro2001 的 ModelScope W4A8 包](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend)，不是 DeepSeek 原始未量化权重。运行时代码取自同一发布方的 [A2 仓库](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B)。模型、代码、基础镜像各有自己的许可与发布责任；复用前自行核对。

## 0. 先确定范围和验收口径

| 项目 | 本包已验证值 |
| --- | --- |
| 平台 | 公有云 ModelArts Standard，**新版**实时推理，专属资源池 |
| 设备 | 每副本一台 8 × Ascend A2/Snt9b；四副本共四台、32 NPU |
| 拓扑 | **每副本本地 TP8/EP8/DP1**；四副本由 ModelArts 在服务入口分流，不是一个跨节点 DP4 引擎 |
| 量化 | 发布方 W4A8 权重、Engram INT8 host-resident |
| 上下文 | 1,048,576 token |
| C5 起点 | batch-token budget 8192、max seq 32、NPU memory 0.92、prefix cache on、spec off、reasoning low |
| 传输 | 容器 HTTP:8000；ModelArts 对外 HTTPS/API key 或项目批准的私网访问 |
| 自动恢复 | 初次部署保持自动重建、故障自动重启关闭，先由人确认故障和回滚 |

保留模型 1M 能力不等于 128K×1K、并发 10 自动达到某个 TTFT/TPOT。功能通过后需另做同条件容量、性能和质量测试。CPU/PLE KV offload 均不在本方案中启用；**Engram 放在 host** 与 KV offload 是不同机制。

部署前填写：区域/项目、专属资源池、8 卡 A2 规格、SFS Turbo 文件系统与挂载网络、私有 SWR 地址、API key 管理方式、日志留存和回滚责任人。至少备妥四台空闲 8 卡节点及充足 SFS 容量。以下命令中的 `CHANGE_ME_*` 必须替换，切勿把实际密钥写进仓库。

## 1. 获取本部署包和发布方源代码

在可访问 ModelScope、GitHub、SWR 的 **ARM64/aarch64 构建 ECS** 上：

```bash
git clone https://github.com/qiuhanzhang616-arch/deepseek-v4.1-flash-modelarts.git modelarts-guide
cd modelarts-guide
git checkout docs/w4a8-modelarts-standard-20260929
# 只有这套 W4A8 包合并到 main 后，才可省略上一行。

git clone https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B.git publisher-a2
git -C publisher-a2 checkout 7e902a1ab49ee299d37bd38c8cde3f680ce4dc7f
git -C publisher-a2 rev-parse HEAD
bash publisher-a2/tools/check_checksums.sh
bash publisher-a2/tools/selfcheck_pkg.sh
```

这里 pin 的源码 commit 是本包验证所用版本。不要把当前 `main`、W8A8 nightly 或 A3 的脚本直接替换进去。发布方 [A2 README](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/a2/README.md) 可作背景资料；本指南的 ModelArts 容器入口与发布方裸机 `docker run` 入口不同：**ModelArts 已提供容器与 NPU，不在容器内再启动 Docker。**

## 2. 下载 W4A8 权重到 SFS Turbo

权重入口：[ModelScope：chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend)。本包 pin 的快照 revision 是 `fa598df636ddc2f9ba4168459effa2875cb0c959`。历史 manifest 为 **120 文件、525,795,301,377 字节**；重组 Engram 时还需要临时空间，推荐至少 1 TiB 的可用 SFS 容量并留日志余量。新 revision 可能改变体积和文件清单，不能沿用历史数字作验收。

在与 ModelArts 专属资源池网络互通的准备 ECS 上挂载**同一个** SFS Turbo，并以该挂载创建目录：

```bash
export SFS_MOUNT=/mnt/CHANGE_ME_SFS_TURBO
export W4A8_ROOT="$SFS_MOUNT/deepseek-v41-w4a8"
mkdir -p "$W4A8_ROOT/weights" "$W4A8_ROOT/startup" "$W4A8_ROOT/results/runtime-logs"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install modelscope requests
python w4a8/tools/download_weights.py --local-dir "$W4A8_ROOT/weights" --workers 8
```

下载脚本使用 ModelScope 的 [`snapshot_download` 及 `revision/local_dir` 参数](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)。下载中断后在同一路径重试；不能以“目录存在”代表快照完整。

### 重组和验算 Engram

发布包把两个巨大 Engram tensor 各拆成 6 片。先把本包的安全重组脚本放到 **权重目录自身**；它严格按 `00`–`05` 的分片名检查 SHA、验证重组结果，并且保留分片。不要从本仓 `tools/` 直接运行（脚本以自身目录为工作目录）：

```bash
if [[ -f "$W4A8_ROOT/weights/engram_int8/reassemble_engram_weights.sh" ]]; then
  cp -n "$W4A8_ROOT/weights/engram_int8/reassemble_engram_weights.sh" \
    "$W4A8_ROOT/weights/engram_int8/reassemble_engram_weights.publisher-original.sh"
fi
install -m 0755 w4a8/tools/reassemble_engram_weights.sh \
  "$W4A8_ROOT/weights/engram_int8/reassemble_engram_weights.sh"
bash "$W4A8_ROOT/weights/engram_int8/reassemble_engram_weights.sh"
python w4a8/tools/verify_weights.py "$W4A8_ROOT/weights" --full-hash
```

期望至少：主权重 72 片、MTPQ 4 片、索引及 `config.json`/tokenizer/量化描述齐全、无 `.incomplete` 文件、两张 Engram weight 与两张 scale 均符合 `PARTS.sha256`。完整 SHA256 会读取数百 GB，**只在准备阶段做一次**并保存日志；Pod 启动时的默认 `ENGRAM_VERIFY=size` 只核大小，不能冒称每个 Pod 重算了 SHA。本包重组脚本不提供 `--delete-parts`，成功后仍保留原始分片。

## 3. 构建 ARM64 W4A8 镜像并推送私有 SWR

本方案**没有可通用直接下载的 ModelArts 成品镜像**。从发布方源码和其 A2 基础镜像构建，然后推送到你项目的私有 SWR。验证过的基础镜像 OCI 引用：

```text
quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler@sha256:0713ca300d55cd907a8beb9fca2444d8f2cf18036e9652f73a636d3dc54fc2a4
```

这是第三方镜像镜像站引用；若在你的网络不可达，先取得等价基础镜像并独立验证 architecture、CANN/vLLM 版本与 digest，**不要只换 tag 就继续部署**。华为云[新版 ModelArts 镜像规范](https://support.huaweicloud.com/intl/en-us/inference-modelarts/inference2.0-modelarts-0003.html)要求把镜像放入 SWR，容器提供健康 API 并将日志写到 stdout。

在 SWR 控制台创建组织/仓库，使用控制台生成的**临时登录命令**完成 `docker login`；凭据不要写在脚本、shell history 或 Git 中。随后：

```bash
export SOURCE_DIR="$PWD/publisher-a2"
export DEST_IMAGE='CHANGE_ME_SWR_REGISTRY/CHANGE_ME_ORG/vllm-ascend:deepseek-v41-flash-w4a8-a2-pinned'
export BUILDER=default
bash w4a8/tools/build_and_push.sh 2>&1 | tee build-w4a8.log
docker buildx imagetools inspect "$DEST_IMAGE"
```

构建脚本固定发布方 commit、基础镜像 digest、`linux/arm64`、`SKIP_PGO=1`，先跑 `check_checksums.sh` 与 `selfcheck_pkg.sh`，再 build/push。记下 SWR 的 **manifest digest**；部署时尽量用不可变 digest，避免同名 tag 被重推。构建机应有足够本地盘/BuildKit 缓存；不要清理他人的 Docker 缓存，也不要把 500+ GB 权重打进镜像。

## 4. 打包并上传 ModelArts 启动脚本

此目录下 `startup/` 是已在 A2 ModelArts Standard 上启动成功的 C5 组合；其中 `clone3_enosys_launcher.c5-nospec.py` 只在宿主策略把 `clone3` 返回 `EPERM`、glibc 无法自动回退时加一层**仍禁止 clone3、但返回 ENOSYS** 的 seccomp 过滤器，原策略没有被关闭。若另一区域/镜像已返回 `ENOSYS` 或线程可正常创建，应先验证是否需要兼容入口，不能盲目套用此过滤器。

```bash
(cd w4a8/startup && sha256sum -c SHA256SUMS)
cp -a w4a8/startup/. "$W4A8_ROOT/startup/"
(cd "$W4A8_ROOT/startup" && sha256sum -c SHA256SUMS)
python -m py_compile "$W4A8_ROOT/startup/clone3_enosys_launcher.c5-nospec.py" \
  "$W4A8_ROOT/startup/c4/sitecustomize.py"
bash -n "$W4A8_ROOT/startup/start_modelarts_single_replica.c4-clone3-otel-stderr.sh"
ln -s weights "$W4A8_ROOT/model-view"
```

最后一行只在 `model-view` **不存在**时执行；如已有对象，先检查目标，不得覆盖。最终 SFS 目录应是：

```text
deepseek-v41-w4a8/
├── weights/              # ModelScope 完整快照 + 已重组的 Engram
├── model-view -> weights # 容器内 MODEL_PATH=/model/w4a8/model-view
├── startup/              # 本包五个经验证入口/配置/遥测文件
└── results/runtime-logs/ # 另挂读写
```

可复现的小包由 `python w4a8/tools/make_bundle.py` 生成；仓库同时提供[下载用 tar.gz](../dist/w4a8-modelarts-standard-20260929.tar.gz)和[SHA256SUMS](../dist/SHA256SUMS)。它仅含本文档与脚本，**不含权重、镜像或密钥**。在准备 ECS 上先校验哈希再解压；解压后也可从此节开始部署。

## 5. 在 ModelArts Standard 新版实时推理创建隔离服务

根据华为云[新版实时推理流程](https://support.huaweicloud.com/intl/en-us/helppanel-modelarts/ma_help_012.html)，先确认专属资源池和 SFS Turbo 都在目标区域，ModelArts agency 能读取私有 SWR 镜像与 SFS，并且资源池网络能访问挂载。**不要进入旧版“创建模型/自定义引擎”流程；新版容器用户与挂载规则不同。**

控制台：**ModelArts → 模型推理 → 实时推理（新版）→ 部署**。逐项填写：

1. 新建隔离服务；选择准备好的专属资源池。服务级按项目安全规范设 API key、HTTPS/私网访问。不要把密钥放在仓库或请求日志。
2. 部署方式用 **基础模式、一个 role-0 单元**；部署副本数 **4**，每副本规格为一台 8 × A2/Snt9b、192 vCPU、约 1.5 TiB RAM（或你的区域里通过验收的等价规格）。四副本各自独立 TP8/EP8/DP1。
3. 选择刚推到私有 SWR 的 W4A8 专用 ARM64 镜像。若平台只允许 tag，部署记录必须同时保存 SWR manifest digest。
4. 将 SFS 的 `/deepseek-v41-w4a8/` **只读**挂到容器 `/model/w4a8`；把同 FS 的 `/deepseek-v41-w4a8/results/` **读写**挂到 `/model/w4a8-results`。两处都选同一 SFS 文件系统，且不要开启本地存储加速。确认容器可解析 `/model/w4a8/model-view/config.json`。
5. 启动命令设为 `python -u /model/w4a8/startup/clone3_enosys_launcher.c5-nospec.py`。脚本先检查兼容性、线程、权重清单和 8 卡，再由镜像里的 `/opt/dsv41/scripts/serve_v2.sh` 启动，不再跑 nested Docker。
6. 容器协议/端口：**HTTP / 8000**。启动探针：HTTP `/health`，周期 5 s、初始延迟 30 s、超时 5 s、失败阈值 720。模型首次加载与静态编译可能耗时数十分钟；尚未监听端口期间的 `connection refused` 不自动等于引擎故障。
7. 首次候选关闭自动重建与故障自动重启，避免重复失败覆盖原始证据。部署超时应覆盖完整权重加载/编译窗口；本次验证环境用 60 min。上线前审阅请求体大小、网关超时、日志留存和回滚配置。
8. 提交前复核 **四副本、同镜像、同挂载、同启动脚本、HTTP:8000、无自动重建**。启动后只读观察事件/日志；不要在加载中叠加新版本。

运行日志应出现 `ctx=1048576 seqs=32 bat=8192 gpu_util=0.92 prefix=1 ... draft_graph=0`、`tp=8 dp=1 ... spec=0`、8 个 rank 已连接、`Application startup complete`、`GET /health ... 200`。Engram size gate 通过后还应检查 host-resident 路径，不能仅凭文件名推断。

## 6. 分层验收

1. ModelArts 服务为 **Running**，部署 **4/4 Ready**，四个 Pod 分布在四台 8 卡 A2 节点；流量权重按预期设置。
2. 各 Pod 无 rank/HCCL/OOM/Engine 死亡、线程创建失败、重复重启；`/health` 200。Pod 的 CPU/内存/NPU 状态与实际启动参数一致。
3. 使用服务“调用信息”里的**精确 URL**和保存在密钥管理器里的 API key，先在安全客户端设临时环境变量：

   ```bash
   export MODELARTS_API_KEY='从密钥管理器临时读取，不写入文件'
   python w4a8/tools/smoke_test.py \
     --base-url 'https://CHANGE_ME_MODELARTS_CALL_URL/v1' \
     --model deepseek-v4.1-flash
   unset MODELARTS_API_KEY
   ```

   `smoke_test.py` 逐层验 `/models`、非流式正确正文和流式 SSE `[DONE]`；如果项目要求 Tool Calling/JSON/视觉，再按客户真实协议补测，不从模型名推断支持。
4. 最后才做正式业务压测和编码质量测试，冻结输入/输出、并发、缓存状态、reasoning、超时及 retry/fallback 口径。**健康/短回答通过不是 128K×1K C10 性能达标**。保留每请求原始结果；HTTP 200 空正文或无 `[DONE]` 记失败。

## 7. 常见故障与回滚

| 现象 | 首先检查 |
| --- | --- |
| `can't start new thread`、OTel 导入失败 | 容器/父级 pids、seccomp 的 `clone3` 返回码、glibc 版本、`sitecustomize` SHA；不要靠降低 batch/seq 掩盖。兼容脚本保持已有 seccomp，未关闭安全策略。 |
| 启动探针 `connection refused` | 权重 78 片加载、静态 kernel 编译是否仍在前进；若停止推进再查 SFS 吞吐、错误和 Pod 事件。 |
| Engram/权重缺片或量化错误 | 停止部署，核对 pinned revision、72+4 shard、四个 Engram tensor/scale 的一次性全 SHA；不要关 Engram 冒充成功。 |
| 模型 API 200 但流式无 `[DONE]` | 分别比较容器直连、ModelArts 前端、代理路径；保留原始 SSE/时间线，不仅看 access log 的 HTTP 200。 |
| 跨节点 HCCL 或无 8 rank | 这通常说明拓扑选错；本包每个 Pod 只用一台本地 8 卡，不是跨节点 TP16/TP32。 |

回滚：保留上一个已验证的部署版本与镜像 digest，按 ModelArts 正规“停止/升级”流程撤回失败版本、恢复原部署/副本数与流量；先核验 `Running`、全部副本 Ready 和 API 语义，再宣告回滚完成。不要删除 SFS 权重、SWR 镜像或 benchmark 原始数据。

## 来源与验证边界

- [W4A8 发布方 ModelScope 权重页](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend)；[发布方 A2 源码](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B)。
- [ModelScope 下载 API](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)。
- [华为云新版实时推理](https://support.huaweicloud.com/intl/en-us/helppanel-modelarts/ma_help_012.html)与[自定义推理镜像规范](https://support.huaweicloud.com/intl/en-us/inference-modelarts/inference2.0-modelarts-0003.html)。
- `startup/` 文件按既有 A2 实测 C5 配置保留字节级版本；变更任一镜像、CANN、驱动、权重 revision、探针或启动脚本都要重新验证。当前包的源码与 tarball 不包含第三方巨型权重或镜像。
