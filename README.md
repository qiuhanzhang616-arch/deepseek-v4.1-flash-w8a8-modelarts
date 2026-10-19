# DeepSeek-V4.1-Flash on Huawei Cloud ModelArts

Production-oriented, reusable deployment package for serving the Ascend W8A8
checkpoint of DeepSeek-V4.1-Flash on **Huawei Cloud public cloud ModelArts
real-time inference (new version)**.

The package adapts the validated vLLM Ascend four-node A2 topology to the
ModelArts multi-unit lifecycle. ModelArts supplies containers, NPU devices,
`POD_IP`, and the global rank table. The scripts discover the rank topology,
verify immutable artifacts, start vLLM, and record operational telemetry.

No tenant address, project ID, resource ID, API key, or registry credential is
included. Values that belong to the deployer's cloud project are explicit
`CHANGE_ME_*` placeholders.

## Validated scope

- **Model:** `Eco-Tech/DeepSeek-V4.1-Flash-w8a8`
- **Pinned revision:** `44201f6d0e14cbc5e73067029762676557a3e7e9`
- **Hardware:** 4 × Atlas 800 A2 nodes, 8 NPUs per node, 32 NPUs total
- **Topology:** DP4 / TP8 / EP32, Prefill and Decode colocated
- **Context:** 1,048,576 tokens
- **Quantization:** Ascend W8A8, INT8 Engram storage
- **Speculative decoding:** DSpark, 5 draft tokens, eager drafter
- **Graph mode:** `FULL_DECODE_ONLY`
- **API:** OpenAI-compatible vLLM API on port 8000
- **Current measured baseline:** `max_num_seqs=32`,
  `max_num_batched_tokens=4096`, `gpu_memory_utilization=0.94`

This is a deployment baseline, not a universal performance promise. Formal
latency, throughput, quality, and concurrency qualification must be repeated in
the target region and resource pool.

Official upstream references:

- DeepSeek-V4.1-Flash on vLLM Ascend:
  <https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md>
- ModelArts create-service API:
  <https://support.huaweicloud.com/intl/en-us/api-modelarts/CreateInferService.html>
- ModelArts SFS Turbo mounting:
  <https://support.huaweicloud.com/intl/en-us/inference-modelarts/inference2.0-modelarts-0090.html>

## Repository layout

```text
config/   ModelArts JSON template and environment examples
docs/     architecture, permissions, operations, and deployment baseline
iam/      reviewable operator and ModelArts-agency policy examples
scripts/  image mirroring, weight preparation, startup, health, and smoke tests
tests/    unit tests and a synthetic 4×8 ModelArts rank table
```

## 1. Collect the target-project values

Copy the environment example and replace every `CHANGE_ME_*` value:

```bash
cp config/deployment.env.example config/deployment.env
chmod 600 config/deployment.env
${EDITOR:-vi} config/deployment.env
```

Required values:

| Variable | Description |
|---|---|
| `MODELARTS_ENDPOINT` | Regional public-cloud ModelArts API endpoint |
| `HUAWEICLOUD_PROJECT_ID` | IAM project ID for the selected region |
| `WORKSPACE_ID` | ModelArts workspace ID, or `0` for default |
| `POOL_ID` | Dedicated ModelArts inference resource pool ID |
| `A2_8_CARD_FLAVOR` | Region-specific 8-NPU A2 flavor ID |
| `SWR_IMAGE_URI` | Private target SWR image URI |
| `SFS_TURBO_ID` | SFS Turbo file-system ID |
| `SFS_SUB_PATH` | Directory containing this package and the weights |

See [permissions and addresses](docs/permissions.md) before continuing.

## 2. Configure IAM and the ModelArts agency

1. Ask the IAM administrator to grant the deployment operator either the
   system-defined `ModelArtsXInferAllPolicy` or a reviewed custom equivalent.
2. In **ModelArts > Permission Management**, create or select a ModelArts
   agency.
3. Grant that agency permission to pull the private SWR image and inspect the
   selected SFS Turbo and dedicated resource pool.
4. If LTS collection is enabled, retain the LTS permissions.

Example policy documents are in [`iam/`](iam/). They are starting points, not
a substitute for the target organization's IAM review.

## 3. Prepare the dedicated resource pool and SFS Turbo

The target pool must contain at least four free nodes, each exposing eight A2
NPUs and sufficient host CPU/RAM for the selected flavor.

1. Create or select a **dedicated inference resource pool** in the target
   region.
2. Create an SFS Turbo file system in the same region with at least 1 TiB of
   available capacity; additional headroom is recommended.
3. Associate the dedicated pool network with SFS Turbo before deployment.
4. Confirm that the pool network CIDRs do not overlap the VPC, SFS Turbo,
   container, service, or reserved inference CIDRs.
5. Mount the SFS Turbo file system on a temporary preparation ECS or ModelArts
   environment in the same VPC.

SFS Turbo mounting is supported only with a dedicated resource pool. Do not
enable local storage acceleration for this baseline.

## 4. Place this package on SFS Turbo

On the preparation host, set a tenant-owned SFS path and clone this repository:

```bash
export DEPLOY_ROOT=/mnt/sfs-turbo/CHANGE_ME_SFS_DIRECTORY
git clone https://github.com/qiuhanzhang616-arch/deepseek-v4.1-flash-w8a8-modelarts.git "$DEPLOY_ROOT"
cd "$DEPLOY_ROOT"
mkdir -p weights manifests runtime-logs benchmark-results
```

The ModelArts template mounts `SFS_SUB_PATH` to
`/model/deepseek-v41-flash/`, so these paths become:

```text
/model/deepseek-v41-flash/weights
/model/deepseek-v41-flash/scripts
/model/deepseek-v41-flash/manifests
/model/deepseek-v41-flash/runtime-logs
```

## 5. Mirror the pinned ARM64 image to private SWR

Pinned upstream image:

```text
quay.io/ascend/vllm-ascend:deepseek-v4.1-flash@sha256:521f866a8d45b5af40cde09c46664838dae41031f4a0ed0183091bdd8ebed2d8
```

Authenticate to the target private SWR using the login command generated by
the SWR console. Do not save the login key in this repository. Then run:

```bash
export DEST_IMAGE='CHANGE_ME_SWR_REGISTRY/CHANGE_ME_ORGANIZATION/CHANGE_ME_REPOSITORY:deepseek-v4.1-flash-521f866a'
bash scripts/mirror_image.sh
```

Copy the resulting private SWR URI into `SWR_IMAGE_URI`. Keep the source digest
and the destination digest in the deployment record.

## 6. Download and verify the checkpoint

The pinned checkpoint is approximately 798 GB. Verify disk quota and network
egress before starting. Install ModelScope in an isolated environment:

```bash
cd "$DEPLOY_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip modelscope
python scripts/download_weights.py --output "$DEPLOY_ROOT/weights" --workers 8
```

The downloader is resumable and writes directly into `weights/`. Do not deploy
while `.incomplete` files exist.

Run the one-time full integrity pass. It hashes the 278 pinned model objects
and verifies the expected 797,570,012,196 bytes. This is intentionally slow:

```bash
python scripts/build_manifests.py --root "$DEPLOY_ROOT"
```

Required outputs:

```text
manifests/weights-sha256-verification.json
manifests/package.sha256
```

Every ModelArts pod checks these reports before loading weights. Do not edit a
script or configuration after building `package.sha256`; regenerate the
manifest after any intentional change.

## 7. Render and review the ModelArts payload

```bash
cd "$DEPLOY_ROOT"
set -a
source config/deployment.env
set +a
python scripts/render_modelarts_config.py \
  --template config/modelarts-service.template.json \
  --output rendered/modelarts-service.json
python -m json.tool rendered/modelarts-service.json >/dev/null
```

Review the rendered file. It must show:

- one `role-0` unit with one 8-NPU A2 instance;
- one `role-1` unit with three 8-NPU A2 instances;
- the same private SWR image, SFS mount, environment, and command on both units;
- port 8000, DP RPC port 13399, and the DP4/TP8/EP32 startup contract;
- API key authentication and no public access unless explicitly approved.

## 8. Deploy through the console or API

Before creating the service, create and securely store the API key described in
Step 10. Binding occurs after the service ID exists.

### Console path

1. Open **ModelArts > Model Inference > Real-Time Inference** (new version).
2. Choose **Deploy** and create a synchronous real-time service.
3. Select the dedicated inference pool.
4. Choose multi-node/multi-PU deployment and create two inference units:
   `role-0` with one instance and `role-1` with three instances.
5. For both units, select the target 8-NPU A2 flavor and private SWR image.
6. Mount the same SFS Turbo subdirectory read/write at
   `/model/deepseek-v41-flash/`; leave local acceleration disabled.
7. Use `bash /model/deepseek-v41-flash/scripts/start.sh` as the command.
8. Copy the environment variables from the rendered JSON to both units.
9. Configure service port 8000, HTTPS, and API key authentication.
10. Set request size to 50 MB, request timeout to 1,200 seconds, and deployment
    timeout to 120 minutes.
11. Leave platform health probes and automatic rebuild disabled for the first
    deployment. The model can spend a long time loading weights and capturing
    graphs; run the supplied health gate after startup instead.
12. Keep public access disabled unless the project security owner explicitly
    approves it. Use private access or an approved ELB for production.

### API path

Export a short-lived IAM token only in the current shell:

```bash
export X_AUTH_TOKEN='CHANGE_ME_SHORT_LIVED_TOKEN'
bash scripts/create_service.sh rendered/modelarts-service.json
unset X_AUTH_TOKEN
```

The API request is sent to:

```text
POST ${MODELARTS_ENDPOINT}/v2/${HUAWEICLOUD_PROJECT_ID}/services
```

Save the returned service ID and version ID in the deployment record.

## 9. Wait for all ranks and run the health gate

ModelArts may start the four pods concurrently. Nodes 1–3 wait up to 3,600
seconds for the node-0 DP RPC endpoint. Do not send traffic merely because the
service object exists.

Check every pod's startup log in `runtime-logs/`. After all engines are ready,
run inside each pod:

```bash
bash /model/deepseek-v41-flash/scripts/health.sh
```

Node 0 must return HTTP 200 from `/health`; each headless node must have a
running `vllm serve --headless` process.

## 10. Create and bind an API key

Create the API key before deploying, because API key authentication is selected
in the service definition. Bind it after the service exists. In **ModelArts >
Model Inference > Real-Time Inference > API Key Authorization Management**:

1. Before deployment, create a key with **Specified real-time services** scope.
2. Download it once and store it in the approved secret manager.
3. After deployment, bind it to the new service.
4. Copy the exact service call URL from **Basic Information > Call Info**.

Do not commit the key. For a smoke test, pass it as an argument from a secret
manager or an ephemeral environment variable:

```bash
python scripts/smoke_test.py \
  --base-url 'CHANGE_ME_CALL_URL' \
  --api-key "$MODELARTS_API_KEY"
```

## 11. Acceptance checklist

- [ ] All four nodes resolve exactly one rank from the ModelArts rank table.
- [ ] All four nodes report 8 visible NPUs.
- [ ] The checkpoint and package manifests pass on every pod.
- [ ] Startup logs agree on `validated-a2-1m-m094`, 1M context, 32 sequences,
      4,096 batched tokens, and 0.94 memory fraction.
- [ ] Four DP engines are ready with TP8 and expert parallelism enabled.
- [ ] `/health` returns 200 and `/v1/models` lists
      `deepseek-v4.1-flash`.
- [ ] Non-streaming and streaming smoke tests pass.
- [ ] Reasoning and tool-call behavior is validated with the target client.
- [ ] No OOM, HCCL error, rank mismatch, restart loop, invalid SSE, or empty
      response occurs.
- [ ] API key, private access, logging, retention, and alerting are approved.
- [ ] A rollback to the recorded image/configuration version is verified.

## 12. Limits and operational guidance

- The ModelArts synchronous path has a fixed 3,600-second end-to-end limit,
  even while streaming. The 1,200-second request-timeout setting does not
  remove that limit.
- The 1M context window is the sum of input and reserved output tokens.
- A directional test completed 833,776 input plus 204,800 output tokens in
  3,488.7 seconds, leaving only 111 seconds before the platform limit. Do not
  market 200K output as a universal production guarantee.
- Do not lower the context length to hide an HBM admission failure if 1M is a
  requirement. Check software equivalence, sidecar overhead, and per-card HBM.
- Do not alter DP/TP/EP, quantization, DSpark depth, graph mode, or the 0.94
  memory fraction without a separately controlled deployment and test plan.
- Formal performance and quality testing are outside this deployment guide.

See [operations and rollback](docs/operations.md) for monitoring and recovery.

## Local package validation

Before publishing or changing the package:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/validate_repository.py .
bash -n scripts/*.sh
python -m json.tool iam/operator-policy.json >/dev/null
python -m json.tool iam/modelarts-agency-policy.json >/dev/null
```
