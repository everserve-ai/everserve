# The engine's image

Every pod of an instance rendered by `integrations/dynamo` runs this image: the head (Dynamo's own vLLM worker with Everserve's executor), the engine's other nodes, and its spare nodes (`everserve_vllm.node_main`).

It contains:

- NVIDIA Dynamo's vLLM runtime, built from Dynamo's sources at the commit in `DYNAMO_PIN` with Dynamo's own renderer (`container/render.py --framework vllm --target runtime`). That runtime is built on `vllm/vllm-openai:v0.31.0`, the vLLM the adapter pins (`adapters/vllm/PIN`). Released Dynamo (1.5.1) pins vLLM 0.28.0, so the pin is a main commit until a release pins 0.31.0 or later.
- `everserve`, `everserve-vllm` and `everserve-dynamo`, installed with the runtime's torch and vLLM held as they are.
- The Everserve kernels, built ahead of time against the runtime's own torch. The build uses the CUDA toolkit in a stage of its own, and the image that runs carries no compiler. A process loads the build only when its stamp (torch, CUDA, the sources) is the process's own (`everserve.kernels`); otherwise it builds the kernels itself or takes its torch paths. The architectures are sm80 (A100), sm89 (L40S), sm90 (H100, H200) and sm100 (B200), set by `TORCH_CUDA_ARCH_LIST`; a GPU outside them builds its own or takes its torch paths.

Before the image can be pushed, its last step checks four things:
- vLLM is 0.31.0;
- every module a pod runs imports;
- the adapter's flags for a Dynamo head render;
- the kernels' stamp matches the image's torch.

## Building

On any linux/amd64 Docker host, from the repository's root:

```
deploy/images/engine/build.sh --tag <registry>/everserve-engine:<tag> [--push]
```

The first build compiles Dynamo's runtime (its Rust wheels, UCX and NIXL) and, with `--push`, keeps it in the same registry as `dynamo-vllm-runtime:<pin>`. A later build at the same pin pulls it from there.

The harness builds it on a rented Nebius CPU VM and releases the VM when the build ends (`bench/harness/build_images.sh --engine`).
