#!/usr/bin/env python3
# =============================================================================
# convert_yolo.py — Convert YOLOv8n PyTorch model to TFLite / ONNX
# =============================================================================
# Purpose:  Provide conversion scripts for deploying YOLOv8 detection models
#           on resource-constrained platforms (Pi 5) using TFLite or ONNX
#           runtimes with INT8 quantisation.
#
# Usage:
#   python tools/convert_yolo.py --weights yolov8n.pt --format tflite
#   python tools/convert_yolo.py --weights yolov8n.pt --format onnx
#   python tools/convert_yolo.py --weights yolov8n.pt --format both
#
# Requirements:
#   pip install ultralytics  (for conversion — NOT needed at runtime)
# =============================================================================
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_to_onnx(weights: Path, imgsz: int = 640) -> Path:
    """Export YOLOv8 model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(weights))
    output = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=13)
    logger.info("ONNX model exported: %s", output)
    return Path(output)


def convert_to_tflite(weights: Path, imgsz: int = 640, int8: bool = True) -> Path:
    """Export YOLOv8 model to TFLite format with optional INT8 quantisation."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(weights))
    output = model.export(format="tflite", imgsz=imgsz, int8=int8)
    logger.info("TFLite model exported: %s", output)
    return Path(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 model to TFLite / ONNX for edge deployment"
    )
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 .pt weights")
    parser.add_argument("--format", default="both", choices=["tflite", "onnx", "both"])
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 quantisation for TFLite")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        logger.error("Weights file not found: %s", weights)
        sys.exit(1)

    if args.format in ("onnx", "both"):
        convert_to_onnx(weights, args.imgsz)

    if args.format in ("tflite", "both"):
        convert_to_tflite(weights, args.imgsz, int8=not args.no_int8)

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
