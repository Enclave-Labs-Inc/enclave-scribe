"""Evaluate model against a ground-truth JSONL using Normalized Edit Distance."""
import argparse
import json
from pathlib import Path


def ned(pred: str, gt: str) -> float:
    try:
        import editdistance
        return editdistance.eval(pred, gt) / max(len(gt), 1)
    except ImportError:
        raise SystemExit("Install editdistance: pip install editdistance")


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe evaluation")
    parser.add_argument("--gt_jsonl", required=True, help="Ground truth JSONL (image, text)")
    parser.add_argument("--model_dir", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_root", default="")
    args = parser.parse_args()

    from scribe.infer.local import infer_image
    from scribe.model.vlm import Qwen2VLModel

    model = Qwen2VLModel()
    model.load(args.model_dir)

    samples = [json.loads(l) for l in open(args.gt_jsonl) if l.strip()]
    scores = []
    for sample in samples:
        image_path = str(Path(args.image_root) / sample["image"]) if args.image_root else sample["image"]
        pred = infer_image(model, image_path)
        score = ned(pred, sample["text"])
        scores.append(score)
        print(f"{sample['image']}: NED={score:.4f}")

    print(f"\nAverage NED: {sum(scores) / len(scores):.4f} over {len(scores)} samples")


if __name__ == "__main__":
    main()
