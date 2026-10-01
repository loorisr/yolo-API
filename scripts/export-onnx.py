#!/usr/bin/env python3
# /// script
# dependencies = ["ultralytics", "onnx", "onnxruntime", "numpy"]
# ///
"""Export a YOLO26 checkpoint to end-to-end NMS-free ONNX.

The models bundled in models/ are RAW exports (1,84,8400, needs NMS).
This script produces the e2e variant (1,300,6) from SPEC §3, which the
server auto-detects and serves without the NMS path.

Usage (uv, no pip):
    uv run scripts/export-onnx.py --model yolo26n.pt --imgsz 640 --out models/yolo26n-e2e.onnx
    MODEL_PATH=/models/yolo26n.onnx  # then bake/point at the e2e file
"""
import argparse
import shutil

import numpy as np
import onnxruntime as ort


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt", help="checkpoint (.pt, auto-downloaded)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--out", default="models/yolo26n-e2e.onnx")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    exported = model.export(
        format="onnx", end2end=True, imgsz=args.imgsz, dynamic=False, opset=args.opset
    )
    print("exported:", exported)
    shutil.move(str(exported), args.out)

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
    x = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    print("sample output shape:", out.shape)
    assert out.shape[1:] == (300, 6), f"not an e2e export: {out.shape}"
    print("OK:", args.out)


if __name__ == "__main__":
    main()
