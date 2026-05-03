#!/usr/bin/env python3
"""Domain shift quantification between train and test sets for reviewer Q5."""
import argparse
import json
import os
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm


def compute_image_stats(img_dir, annotations, images_dict, max_samples=2000):
    """Compute color statistics for a set of ROI images."""
    means_r, means_g, means_b = [], [], []
    stds_r, stds_g, stds_b = [], [], []
    widths, heights, aspects = [], [], []

    for ann in tqdm(annotations[:max_samples], desc="Processing images"):
        img_id = ann['image_id']
        ann_id = ann['id']
        img_path = os.path.join(img_dir, '1x', f"{img_id}_{ann_id}.png")
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspects.append(w / max(h, 1))
            means_r.append(arr[:, :, 0].mean())
            means_g.append(arr[:, :, 1].mean())
            means_b.append(arr[:, :, 2].mean())
            stds_r.append(arr[:, :, 0].std())
            stds_g.append(arr[:, :, 1].std())
            stds_b.append(arr[:, :, 2].std())
        except Exception:
            continue

    return {
        'n': len(widths),
        'width': {'mean': np.mean(widths), 'std': np.std(widths), 'median': np.median(widths)},
        'height': {'mean': np.mean(heights), 'std': np.std(heights), 'median': np.median(heights)},
        'aspect_ratio': {'mean': np.mean(aspects), 'std': np.std(aspects)},
        'color_mean': {
            'R': np.mean(means_r), 'G': np.mean(means_g), 'B': np.mean(means_b),
        },
        'color_std': {
            'R': np.mean(stds_r), 'G': np.mean(stds_g), 'B': np.mean(stds_b),
        },
        'brightness': {'mean': np.mean([np.mean([r, g, b]) for r, g, b in zip(means_r, means_g, means_b)]),
                       'std': np.std([np.mean([r, g, b]) for r, g, b in zip(means_r, means_g, means_b)])},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-coco', default='data/dataset_train.json')
    parser.add_argument('--test-coco', default='data/dataset_test.json')
    parser.add_argument('--train-rois', default='train/rois')
    parser.add_argument('--test-rois', default='test/rois')
    args = parser.parse_args()

    with open(args.train_coco) as f:
        train_data = json.load(f)
    with open(args.test_coco) as f:
        test_data = json.load(f)

    train_images = {img['id']: img for img in train_data['images']}
    test_images = {img['id']: img for img in test_data['images']}

    print(f"Train: {len(train_data['annotations'])} annotations, {len(train_images)} images")
    print(f"Test: {len(test_data['annotations'])} annotations, {len(test_images)} images")

    # Image-level statistics from COCO metadata
    print("\n" + "=" * 70)
    print("Image-Level Statistics (from COCO metadata)")
    print("=" * 70)

    train_w = [img['width'] for img in train_images.values()]
    train_h = [img['height'] for img in train_images.values()]
    test_w = [img['width'] for img in test_images.values()]
    test_h = [img['height'] for img in test_images.values()]

    print(f"\n  Full image resolution:")
    print(f"    Train: {np.mean(train_w):.0f}x{np.mean(train_h):.0f} (mean), "
          f"{np.median(train_w):.0f}x{np.median(train_h):.0f} (median)")
    print(f"    Test:  {np.mean(test_w):.0f}x{np.mean(test_h):.0f} (mean), "
          f"{np.median(test_w):.0f}x{np.median(test_h):.0f} (median)")

    # Bounding box statistics
    train_bbox_w, train_bbox_h = [], []
    for ann in train_data['annotations']:
        if 'bbox' in ann:
            _, _, bw, bh = ann['bbox']
            train_bbox_w.append(bw)
            train_bbox_h.append(bh)

    test_bbox_w, test_bbox_h = [], []
    for ann in test_data['annotations']:
        if 'bbox' in ann:
            _, _, bw, bh = ann['bbox']
            test_bbox_w.append(bw)
            test_bbox_h.append(bh)

    if train_bbox_w and test_bbox_w:
        print(f"\n  Bounding box dimensions:")
        print(f"    Train: {np.mean(train_bbox_w):.1f}x{np.mean(train_bbox_h):.1f} (mean), "
              f"{np.median(train_bbox_w):.0f}x{np.median(train_bbox_h):.0f} (median)")
        print(f"    Test:  {np.mean(test_bbox_w):.1f}x{np.mean(test_bbox_h):.1f} (mean), "
              f"{np.median(test_bbox_w):.0f}x{np.median(test_bbox_h):.0f} (median)")
        print(f"    Scale ratio (test/train): {np.mean(test_bbox_w)/np.mean(train_bbox_w):.2f}x width, "
              f"{np.mean(test_bbox_h)/np.mean(train_bbox_h):.2f}x height")

    # Annotations per image
    train_anns_per_img = defaultdict(int)
    for ann in train_data['annotations']:
        train_anns_per_img[ann['image_id']] += 1
    test_anns_per_img = defaultdict(int)
    for ann in test_data['annotations']:
        test_anns_per_img[ann['image_id']] += 1

    print(f"\n  Annotations per image:")
    print(f"    Train: {np.mean(list(train_anns_per_img.values())):.1f} (mean), "
          f"{np.median(list(train_anns_per_img.values())):.0f} (median), "
          f"max={max(train_anns_per_img.values())}")
    print(f"    Test:  {np.mean(list(test_anns_per_img.values())):.1f} (mean), "
          f"{np.median(list(test_anns_per_img.values())):.0f} (median), "
          f"max={max(test_anns_per_img.values())}")

    # ROI color statistics
    print("\n" + "=" * 70)
    print("ROI Color Statistics (1x crops)")
    print("=" * 70)

    train_stats = compute_image_stats(args.train_rois, train_data['annotations'], train_images, max_samples=3000)
    test_stats = compute_image_stats(args.test_rois, test_data['annotations'], test_images, max_samples=1000)

    print(f"\n  ROI dimensions (pixels after crop):")
    print(f"    Train: {train_stats['width']['mean']:.0f}x{train_stats['height']['mean']:.0f} (mean)")
    print(f"    Test:  {test_stats['width']['mean']:.0f}x{test_stats['height']['mean']:.0f} (mean)")

    print(f"\n  Mean color (RGB, 0-1):")
    print(f"    Train: R={train_stats['color_mean']['R']:.3f}, G={train_stats['color_mean']['G']:.3f}, "
          f"B={train_stats['color_mean']['B']:.3f}")
    print(f"    Test:  R={test_stats['color_mean']['R']:.3f}, G={test_stats['color_mean']['G']:.3f}, "
          f"B={test_stats['color_mean']['B']:.3f}")

    print(f"\n  Color variability (per-image std, mean across set):")
    print(f"    Train: R={train_stats['color_std']['R']:.3f}, G={train_stats['color_std']['G']:.3f}, "
          f"B={train_stats['color_std']['B']:.3f}")
    print(f"    Test:  R={test_stats['color_std']['R']:.3f}, G={test_stats['color_std']['G']:.3f}, "
          f"B={test_stats['color_std']['B']:.3f}")

    print(f"\n  Brightness (mean luminance):")
    print(f"    Train: {train_stats['brightness']['mean']:.3f} ± {train_stats['brightness']['std']:.3f}")
    print(f"    Test:  {test_stats['brightness']['mean']:.3f} ± {test_stats['brightness']['std']:.3f}")

    # Species distribution
    print("\n" + "=" * 70)
    print("Species Distribution Comparison")
    print("=" * 70)

    train_cats = defaultdict(int)
    for ann in train_data['annotations']:
        train_cats[ann['category_id']] += 1

    # Test doesn't have category_id typically, so skip
    train_counts = sorted(train_cats.values())
    print(f"  Train species: {len(train_cats)} unique")
    print(f"  Train samples/species: min={min(train_counts)}, max={max(train_counts)}, "
          f"mean={np.mean(train_counts):.1f}, std={np.std(train_counts):.1f}")

    # Summary for paper
    print(f"\n" + "=" * 70)
    print("LaTeX Summary")
    print("=" * 70)

    # Compute Cohen's d for brightness shift
    d_brightness = abs(train_stats['brightness']['mean'] - test_stats['brightness']['mean']) / \
        np.sqrt((train_stats['brightness']['std']**2 + test_stats['brightness']['std']**2) / 2)
    print(f"  Brightness Cohen's d: {d_brightness:.2f}")

    if train_bbox_w and test_bbox_w:
        d_width = abs(np.mean(train_bbox_w) - np.mean(test_bbox_w)) / \
            np.sqrt((np.std(train_bbox_w)**2 + np.std(test_bbox_w)**2) / 2)
        print(f"  BBox width Cohen's d: {d_width:.2f}")


if __name__ == '__main__':
    main()
