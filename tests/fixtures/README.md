# MOSS graph metadata fixture

`moss-selected-dit-header.json` contains 379 tensor names, dimensions and GGML type IDs, with source identities. It contains no tensor values, tokenizer, prompt, image, credential, raw console log or learned weight payload.

The metadata was extracted from exactly the first 23,320 bytes of `MOSS-Image-SFT-Q4_K.gguf`, pinned to [leejet/MOSS-Image-SFT-GGUF revision 238daf4](https://huggingface.co/leejet/MOSS-Image-SFT-GGUF/tree/238daf423f0de65316207453591f275407aac9a7). The header SHA256 is `823f3c7079baeee5a0168db3446786dda83a0f3251a914bde7d8deb3c85b58f6`; the full model identity is in the fixture and [model lock](../../config/models.lock.json). The model is Apache-2.0; preserve the existing `licenses/MOSS-models-*.txt` notices. This fixture does not replace full model download/hash verification for deployment.

The baseline scratch header is reconstructed by the runner from pinned repository commit `189632386f782fc209e7f3ae1ea6e47138f6475d`; expected SHA256 `95fae94bf8035e175dbad4011a843d1fd4c049179a081a87a751c91db9b4ae73`. It is kept separate from the current production header and is never installed into the engine. A normal full Git clone includes that ancestor; a shallow checkout must provide it before running this offline test.

With the prepared source/dependency paths from [build instructions](../../docs/build.md), run from the repository root:

```powershell
python -B tests/moss_actual_graph_scratch.py --config D:/PS5-AI/moss/build/build-config.private.json --out D:/PS5-AI/moss/host-tests/actual-graph-repro-001
```

Only the config's `LLVM` and `SD` paths are used. The fresh output directory must be outside the repository on D:. The runner opens no model file and makes no network/PS5 connection. It compiles fresh host metadata/quantization objects and copies four exact helper bodies from the prepared source into the temporary build; no CPU graph backend or Vulkan library is linked.

Expected: 1,079 checks; actual 42-layer `Model::forward` constructs graph metadata for 48/45 text tokens and a 16×16×128 latent. The baseline reproduces 17 unknown BF16-left `MUL_MAT` nodes; the corrected bound accepts all 17 within a 1GiB per-claim limit. Noncontiguous BF16, BF16 activations, insufficient claims and unreviewed fusion remain rejected. No graph executes and no tensor data is allocated. The selected runtime LoRA graph is deliberately outside this base-graph test; LoRA bindings and actual GPU generation have separate evidence.
