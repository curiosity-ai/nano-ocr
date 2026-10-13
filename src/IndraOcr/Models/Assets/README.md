# Embedded models

These are the Indra OCR models from the Hugging Face repository
[`datalab-to/indra-ocr`](https://huggingface.co/datalab-to/indra-ocr), tag `v3.1` — the model
set the Python package 0.3.1 (in `.reference/`) was built and tested against. They are compiled
into `IndraOcr.dll` as embedded resources, so the library needs no download at run time.

| File | SHA-256 |
| --- | --- |
| det.onnx | 46b0050ce4c4692f887a1a9c038480223af1c7a91b3fcced2657d9cf00290dc4 |
| rec.onnx | e243be069b357e56a0e867a21d7482e498df42fb19678bfb67a25f48b1f06151 |
| rec_cjk.onnx | 4cec840b55e1bb7f375daf30a2975404a867a71ac1a5bc5b1629c4909ce16baa |
| script.onnx | 5ba37030e817affede0097a7503b6829624104166c6e0b6aacecacf919fc7d11 |
| order.onnx | 23635dd2324b4ccc00c408af961273a2dedd2e4f0386bc54dba678c41bcf6b46 |
| orient.onnx | 8d7bfce5d08bcad55955f41ca494a7f9d9d3ed5d5b6da1f6f276d5c509590729 |
| mathcls.onnx | 818b1556e9ef337958db6be0c9791299d3f74a4575586b0cd469aaae9534ff96 |
| math_enc.onnx | d4e4a6c40dad1ee988d32c6403b1590ca42864d4483bbe1e671c5dbb999a5a97 |
| math_dec_init.onnx | 919417fde9962c3318aeb478c1fe3955dd2bb8cd34476c68debed2ba39e5dcd6 |
| math_dec_step.onnx | 5ff6e6cbaf0dea70314f66222a1469736a8c5e7994fab06a5b4e53c90469239f |

The weights are under Datalab's modified OpenRAIL-M license (`MODEL_LICENSE`), not the
Apache-2.0 license of the code.
