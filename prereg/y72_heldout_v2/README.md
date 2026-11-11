# Y72 held-out v2: what is in this directory

Pre-registered in `../y72_heldout_v2.json` (commit d7e6147). This repository ignores `*.pt` and `*.log`, so two things the registration points at were not in git when it was committed:

* the binary set `y72_heldout_v2.pt` (kept on the generating machine; its file sha256 is registered) -> the same set is committed as `y72_heldout_v2_set.json` (hex int16 little-endian rows and mask plus payloads). Its identity is the registered **canonical sha256** (`16ff53a8...e71b`) and payload-list sha256, which `tools/experiment_packaging/y72_heldout_v2.py::verify_text_set` recomputes fail-closed. Adding the text form changes none of the registered hashes.
* the generation / overlap logs `*.log` -> committed byte-identical as `*.log.txt`.

`overlap_check.json`, `gen_info.json`, `noise_rescale.json` and the three scripts are as registered.
