# DeepSeek-V4.1-Flash W4A8 Production Deployment on ModelArts Standard

This runbook records the currently verified production shape for
`deepseek-v4.1-flash` on Ascend A2. It is specific to the internal W4A8 image
and checkpoint family below. Do not substitute the upstream W8A8 image or
checkpoint, an A3 recipe, or a PD topology without a separate compatibility
and performance qualification.

> **Production snapshot:** read-only route/deployment check on 2026-10-04 at
> 20:41 UTC. The active deployment was version `1.0.7`, 4/4 ready,
> with 100% traffic. The customer-facing model name remained
> `deepseek-v4.1-flash`. Service UUIDs, endpoints, node addresses, API keys,
> and tenant-specific SFS/SWR identifiers are deliberately omitted.

## 1. Architecture and serving contract

| Setting | Production profile |
|---|---|
| Platform | Huawei Cloud ModelArts Standard, real-time inference (new version) |
| Accelerator | Four independent Atlas 800 A2 replicas; one 8-NPU host per replica (32 NPUs total) |
| Parallelism per replica | TP8 / EP8 / DP1; Prefill and Decode colocated |
| Model/API name | `deepseek-v4.1-flash` |
| Context limit | 1,048,576 tokens |
| Scheduler | `max_num_seqs=32`; `max_num_batched_tokens=8192` |
| Memory target | `gpu_memory_utilization=0.92` |
| Weights | Publisher-provided W4A8 DeepSeek-V4.1-Flash checkpoint; verify the exact SFS snapshot before each rollout |
| Speculative decoding | DSpark, 5-token proposal; draft runs eager (`DRAFT_GRAPH=0`, `SPEC_EAGER=1`) |
| Main model execution | `FULL_DECODE_ONLY` graph mode; NPUGraph EX enabled; static-kernel path disabled |
| Engram | INT8, host-resident; automatic device-index selection, fallback disabled |
| Prefix reuse | vLLM prefix caching enabled. This is separate from ModelArts local-storage acceleration. |
| Container listener | HTTP port 8000 inside the ModelArts container |
| External access | Preserve the existing HTTPS and API-key boundary; never put the key in this repository |

Each ModelArts replica is its own 8-card inference engine. This is **not** one
cross-node DP4 engine and it is **not** a Prefill/Decode-disaggregated service.
The external ModelArts/LiteLLM name stays `deepseek-v4.1-flash`.

## 2. Production image and artifact pinning

The verified production image reference shown by ModelArts is:

```text
modelarts/vllm-ascend:deepseek-v41-flash-a2-w4a8-ee0892-20260930
```

This is the image name and tag shown in ModelArts, **not a complete public
`docker pull` address**. The production record does not contain the private SWR
registry hostname/path or OCI manifest digest, so the exact production image
cannot currently be downloaded by another project from this guide. In the
owning project, open the image details in SWR and copy its full pull command
and immutable digest. The full reference should have this form (placeholders
must be replaced with the values from SWR):

```text
swr.<region>.myhuaweicloud.com/<organization>/vllm-ascend:deepseek-v41-flash-a2-w4a8-ee0892-20260930@sha256:<manifest-digest>
```

The destination project also needs authorized image-pull access. Do not guess
the registry hostname from the ModelArts short name above.

### Public build inputs (not the finished production image)

If the private production image cannot be shared, this repository records a
separate ARM64 W4A8 build path. It uses the following base-image address:

```text
quay.nju.edu.cn/ascend/vllm-ascend:deepseek-v4.1-flash-openeuler@sha256:0713ca300d55cd907a8beb9fca2444d8f2cf18036e9652f73a636d3dc54fc2a4
```

This is a third-party mirror of the **base vLLM-Ascend image only**. It is not
the finished W4A8 production image, and it does not include the W4A8 runtime
customizations or checkpoint. If network policy blocks this mirror, use an
approved equivalent base only after checking its architecture, CANN/runtime
versions, and digest.

The build inputs are:

- Pinned publisher A2 source: [`chiro2001/deepseek-v4.1-flash-ascend910B`
  at commit `7e902a1ab49ee299d37bd38c8cde3f680ce4dc7f`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/tree/7e902a1ab49ee299d37bd38c8cde3f680ce4dc7f).
- Build/push steps: [W4A8 ModelArts guide](README.md) and
  [`tools/build_and_push.sh`](tools/build_and_push.sh). The script builds a
  new image from its own pinned publisher source/base and pushes it to a
  destination SWR URI supplied by the project owner.

That script creates a new build; it does **not** claim to reproduce the
production tag byte-for-byte. The tag contains `ee0892`, and an accompanying
runtime-preparation record associates that build label with source commit
`ee0892aa68dd0faa7fc24c31e9690facc58660a2`, but the production record itself
does not include `BUILD_INFO` or a manifest digest to verify the exact binary.
Validate a rebuilt image separately before using it. Do not substitute the
official upstream W8A8 image for this W4A8 image.

The production checkpoint is stored on the approved SFS Turbo model mount.
Keep the mounted model and Engram files read-only. Record the exact publisher
snapshot/revision, file inventory, and one-time SHA-256 manifest in the
deployment change record; the live deployment note does not preserve an
independently verified checkpoint revision or manifest digest.

## 3. Prepare the release on the staging host

1. Use the approved ARM64 staging/build host with network access to the
   publisher, private SWR, and the target SFS Turbo. Do not build or download
   hundreds of gigabytes from inside a ModelArts inference container.
2. Create a new, immutable, date/versioned SFS release directory. Do not edit
   the directory used by the running release.
3. Place the W4A8 checkpoint, DSpark/MTPQ draft artifacts, INT8 Engram tables,
   tokenizer/configuration, and required startup files in their expected
   directories. Confirm all publisher Engram shards are reassembled using the
   publisher-provided checksum procedure.
4. Validate the exact W4A8 file layout and hashes. Save the verification output
   beside the release manifest. Stop if any shard, scale, tokenizer, or Engram
   artifact is missing or mismatched.
5. Copy the approved ModelArts launcher, startup wrapper, environment file,
   and Engram preflight helper into the new release. Keep secrets out of these
   files. Compare hashes with the approved release package before submission.
6. Give each Pod a symlink-only model view that resolves to the same immutable
   read-only model files. Use a unique view path per Pod so concurrent launchers
   cannot race while creating shared symlinks.
7. Separate storage by access pattern: model/Engram mounts are read-only;
   logs, runtime state, and results use a dedicated read-write path. Do not
   make the shared model tree writable to work around an Engram mount error.

## 4. Build the ModelArts service configuration

Use **ModelArts → Model Inference → Real-Time Inference (new version)**. ModelArts
already provides the container and NPUs: do **not** run nested Docker from the
startup wrapper.

1. Select the approved dedicated A2 resource pool. For a parallel isolated
   canary, verify four additional free hosts; for an in-place production
   upgrade, verify that the existing four hosts remain assigned to the service
   and that CPU/RAM are schedulable during replacement. In either case, check
   eight allocatable NPUs per host and SFS reachability. The scheduler may
   reserve CPU even when a flavor is nominally 192 vCPU; check live allocatable
   resources rather than relying on the flavor label alone.
2. Configure four replicas, one 8-NPU A2 instance per replica. Apply strong
   node affinity only to the four approved hosts. Confirm the final placement
   has exactly four hosts and does not concentrate replicas on fewer nodes.
3. Select the exact internal W4A8 image above. Capture both the selected tag
   and its immutable registry digest in the change record. Confirm the runtime
   `BUILD_INFO` matches the intended build.
4. Mount the model release read-only. Set ModelArts **local storage
   acceleration ON** for the model and read-only Engram mounts. This platform
   storage-startup setting is not vLLM prefix caching. Mount logs/state/results
   read-write and uncached so concurrent writers see coherent files.
5. Set the startup command to the approved launcher from the release directory.
   The launcher must resolve Pod IP safely, check for eight visible NPUs, verify
   required files, report effective settings, and then invoke the server.
6. Configure the container listener as **HTTP:8000**; keep external ingress
   HTTPS/API-key protected. Do not publish the internal listener or API key.
7. Configure a health/startup probe for the actual server health path and allow
   enough time for cold SFS loading and graph compilation. Probe tuning must
   not mistake a progressing cold start for a permanently ready engine.
8. Enable graceful shutdown and disable automatic rebuild. Keep the existing
   external security settings, service identity, and client model alias.
9. Review the upgrade policy before submission. A four-replica rollout with
   zero surge can cause a full interruption when all replicas are replaced;
   use an approved maintenance window or an explicitly qualified capacity
   plan. Preserve the previous known-good version and release directory.

## 5. Apply the serving settings

Set and then **verify the effective process arguments** for each replica:

```text
model name                 deepseek-v4.1-flash
model length               1048576
max sequences              32
max batched tokens          8192
NPU memory utilization      0.92
parallelism                 TP=8, EP=8, DP=1
quantization                publisher W4A8 checkpoint
DSpark                      enabled, 5 proposal tokens, draft eager
main-model graph            FULL_DECODE_ONLY
NPUGraph EX                 enabled
static-kernel path          disabled
Engram                      INT8, host-resident
prefix caching              enabled
default reasoning           low
```

The draft-eager setting does **not** mean the main model is fully eager: the
target model still uses the configured decode graph. Do not add a separate
`draft_window_size` override unless the pinned runtime explicitly requires it;
verify the actual resolved DSpark configuration and startup logs instead of
assuming an empty environment variable overrides a default.

## 6. Deploy first without customer traffic

1. Preserve the running production version and create/update the candidate only
   through the normal ModelArts console/API workflow.
2. Prefer an isolated candidate service on separately available capacity. Keep
   it outside the customer route and at zero traffic until the startup gates
   below pass. If no spare four-node capacity exists, do not claim that a
   parallel canary is possible: an in-place rollout needs an approved
   maintenance window and can interrupt all traffic when surge is zero.
3. Check ModelArts events for SFS mount success, image pull, scheduling, and
   restarts. Verify all four replicas are ready on the intended hosts.
4. From LTS/runtime logs, confirm all ranks joined, each process sees eight
   NPUs, the served model name is correct, and effective parameters match the
   table above. Record image `BUILD_INFO`, image digest, startup-script hash,
   weight manifest identity, Pod placement, and deployment version.
5. If any replica repeatedly exits, an artifact check fails, or the effective
   config differs, stop the rollout and preserve logs. Do not hide startup
   failures by routing traffic to another model or fallback.

## 7. Run minimum acceptance checks

After all four replicas are ready:

1. Verify every replica's engine health endpoint and `/v1/models` response.
2. Confirm the only served model ID is `deepseek-v4.1-flash`.
3. Send a minimal non-streaming request and a streaming request through the
   isolated service using an API key injected from the approved secret manager.
   Verify a non-empty response, correct stream termination (`[DONE]`), and no
   unexpected truncation or encoding corruption.
4. Check queue/running-request metrics return to idle and no new health-check
   failure, restart, OOM, or HCCL error appears.
5. Only after the isolated candidate passes—or during the explicitly approved
   in-place maintenance window—perform the production version update while
   preserving HTTPS/API-key ingress and the LiteLLM model name. Verify the
   active route still resolves to this service and the name remains
   `deepseek-v4.1-flash`.

These checks establish availability and interface behavior only. They are not
a throughput, latency, concurrency, task-quality, or capacity qualification.
Run a separate, approved AISBench/customer-shaped matrix with frozen inputs,
outputs, cache state, reasoning, concurrency, retries, and quality gates.

### Production verification snapshot

At the last recorded read-only check (2026-10-04 20:41 UTC), the customer alias
resolved to the W4A8 production service; the active version `1.0.7` was 4/4
ready with 100% traffic. This verifies control-plane
availability and routing only, not performance. Recheck the service and
effective runtime before relying on this snapshot for a later change.

## 8. Logging, data, and secret handling

- Keep external HTTPS and API-key authentication. Read the key from the
  approved secret manager at request time; never commit it or echo it to logs.
- LTS/request logging may retain prompt or response material. Confirm the
  approved retention period and data classification with the service owner.
  The production rollout previously accepted request-bearing logs with a
  30-day retention period; do not assume that approval transfers to a new
  project or longer retention.
- Do not enable full request-body logging for a diagnostic by default. Preserve
  startup/runtime events needed to diagnose launch failures without logging
  credentials or dumping the entire process environment.

## 9. Rollback

1. Keep the prior ModelArts version, startup bundle, image reference, and SFS
   release intact until the new version completes acceptance.
2. If a health, runtime, API, or quality gate fails, keep the new candidate at
   zero traffic and restore the prior deployment version using the supported
   ModelArts rollback/upgrade flow.
3. Restore the previous traffic policy only after the prior version is ready
   and its `/v1/models` identity is confirmed.
4. Verify the LiteLLM name remains `deepseek-v4.1-flash`, HTTPS/API-key
   protection is unchanged, and all replicas are healthy.
5. Preserve failed-version events, logs, hashes, and test output. Do not delete
   the failed release or raw results while investigating it.

## 10. Known evidence gaps

- The production image tag is recorded, but its immutable registry manifest
  digest was not present in the deployment evidence available when this guide
  was written. Obtain it before claiming byte-for-byte reproducibility.
- The private SWR registry hostname, organization path, and cross-project pull
  permission are not present in the deployment record. The tag above is the
  ModelArts selection reference, not a portable public download URL. The
  base-image URL and build inputs are provided as an alternative way to create
  a separately validated image, not to retrieve the exact production bits.
- The exact production weight snapshot/revision and full SFS checksum manifest
  should be linked from the controlled release record; this public guide
  intentionally contains no tenant SFS path or internal endpoint.
- The 2026-10-04 4/4/100% snapshot is a control-plane status check, not a formal
  inference or performance result.

## References

- [Publisher W4A8 ModelScope checkpoint](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-w4a8-Ascend)
- [Publisher A2 runtime repository](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B)
- [ModelArts Standard W4A8 guide in this repository](README.md)
- [Huawei Cloud ModelArts Standard real-time inference](https://support.huaweicloud.com/intl/en-us/helppanel-modelarts/ma_help_012.html)
- [Huawei Cloud custom inference image requirements](https://support.huaweicloud.com/intl/en-us/inference-modelarts/inference2.0-modelarts-0003.html)
