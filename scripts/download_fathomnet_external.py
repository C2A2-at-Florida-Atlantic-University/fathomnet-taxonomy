#!/usr/bin/env python3
"""
Download an external validation dataset from the FathomNet database API.

For each of the 79 species in taxonomy.csv, queries the FathomNet API for images,
excludes any that overlap with the competition train/test sets, downloads the full
images, and crops multi-scale ROIs (1x, 3x, 5x) + full image.

Produces a COCO-format JSON and annotations CSV compatible with the existing
data pipeline and inference scripts.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger("fathomnet_external")
HANDLER = logging.StreamHandler()
HANDLER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDLER)

API_BASE = "https://database.fathomnet.org/api"


def load_competition_urls(train_json: str, test_json: str) -> set:
    """Load all image URLs from competition train/test to exclude."""
    urls = set()
    for path in [train_json, test_json]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for img in data["images"]:
                urls.add(img["coco_url"])
            LOGGER.info(f"Loaded {len(data['images'])} URLs from {path}")
    LOGGER.info(f"Total competition URLs to exclude: {len(urls)}")
    return urls


def load_taxonomy(taxonomy_csv: str) -> list[str]:
    """Load species names from taxonomy CSV."""
    species = []
    with open(taxonomy_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            species.append(row["species"])
    return species


def query_fathomnet_images(concept: str, limit: int = 0) -> list[dict]:
    """Query FathomNet API for images containing a concept."""
    url = f"{API_BASE}/images/query/concept/{concept}"
    try:
        resp = httpx.get(url, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        images = resp.json()
        if limit > 0:
            images = images[:limit]
        return images
    except Exception as e:
        LOGGER.warning(f"Failed to query {concept}: {e}")
        return []


def expand_bbox(bbox, scale, image_width, image_height):
    """Expand a bounding box by a scale factor, centered, clipped to image."""
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    new_w = w * scale
    new_h = h * scale
    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(new_w, image_width - new_x)
    new_h = min(new_h, image_height - new_y)
    return [int(new_x), int(new_y), int(new_w), int(new_h)]


async def download_image(
    client: httpx.AsyncClient, url: str, output_path: Path,
    max_retries: int = 3, backoff: float = 1.5
) -> bool:
    """Download an image with retries. Returns True on success."""
    if output_path.exists():
        return True
    delay = backoff
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.get(url, timeout=60.0, follow_redirects=True)
            resp.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception as e:
            LOGGER.debug(f"[Attempt {attempt}] Failed {url}: {e}")
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= backoff
    LOGGER.warning(f"Failed to download after {max_retries} attempts: {url}")
    return False


def crop_roi(image_path: Path, bbox: list, output_path: Path) -> bool:
    """Crop an ROI from an image and save it."""
    if output_path.exists():
        return True
    try:
        with Image.open(image_path) as img:
            x, y, w, h = bbox
            if w < 2 or h < 2:
                return False
            cropped = img.crop((x, y, x + w, y + h))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path)
            return True
    except Exception as e:
        LOGGER.debug(f"Crop failed for {image_path}: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", default="data/taxonomy.csv",
                        help="Path to taxonomy CSV")
    parser.add_argument("--train-json", default="data/dataset_train.json",
                        help="Competition train JSON (to exclude)")
    parser.add_argument("--test-json", default="data/dataset_test.json",
                        help="Competition test JSON (to exclude)")
    parser.add_argument("--output-dir", default="external_val",
                        help="Output directory")
    parser.add_argument("--max-per-species", type=int, default=50,
                        help="Max annotations per species to download")
    parser.add_argument("--max-images-query", type=int, default=200,
                        help="Max images per API query")
    parser.add_argument("--concurrent", type=int, default=8,
                        help="Max concurrent downloads")
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 3.0, 5.0],
                        help="ROI scale factors")
    parser.add_argument("--verified-only", action="store_true", default=True,
                        help="Only use verified bounding boxes")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    log_level = (logging.DEBUG if args.verbose >= 2
                 else logging.INFO if args.verbose >= 1
                 else logging.WARNING)
    LOGGER.setLevel(log_level)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    rois_dir = output_dir / "rois"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load exclusion set
    exclude_urls = load_competition_urls(args.train_json, args.test_json)

    # Load taxonomy
    species_list = load_taxonomy(args.taxonomy)
    print(f"Querying FathomNet for {len(species_list)} concepts...")

    # Phase 1: Query API for all species and collect image/bbox metadata
    all_images = {}  # url -> image metadata
    annotations = []  # list of (url, bbox_dict, concept_name)
    species_counts = defaultdict(int)

    for concept in tqdm(species_list, desc="Querying API"):
        api_images = query_fathomnet_images(concept, limit=args.max_images_query)

        for img_data in api_images:
            img_url = img_data["url"]

            # Skip competition images
            if img_url in exclude_urls:
                continue

            # Find matching bounding boxes for this concept
            for bb in img_data.get("boundingBoxes", []):
                if bb["concept"] != concept:
                    continue
                if args.verified_only and bb.get("reviewState") != "VERIFIED":
                    continue
                if args.max_per_species > 0 and species_counts[concept] >= args.max_per_species:
                    break

                # Store image metadata
                if img_url not in all_images:
                    all_images[img_url] = {
                        "url": img_url,
                        "width": img_data.get("width", 0),
                        "height": img_data.get("height", 0),
                        "uuid": img_data.get("uuid", ""),
                    }

                annotations.append({
                    "image_url": img_url,
                    "bbox": [bb["x"], bb["y"], bb["width"], bb["height"]],
                    "concept": concept,
                    "bbox_uuid": bb["uuid"],
                })
                species_counts[concept] += 1

            if args.max_per_species > 0 and species_counts[concept] >= args.max_per_species:
                break

    # Summary
    print(f"\nAPI query results:")
    print(f"  Total unique images: {len(all_images)}")
    print(f"  Total annotations: {len(annotations)}")
    print(f"  Species with data: {sum(1 for c in species_counts if species_counts[c] > 0)}/{len(species_list)}")

    missing = [s for s in species_list if species_counts[s] == 0]
    if missing:
        print(f"  Species with 0 results: {missing}")

    low = [(s, c) for s, c in species_counts.items() if 0 < c < 5]
    if low:
        print(f"  Species with <5 results: {[(s, c) for s, c in low]}")

    # Print per-species counts
    print(f"\nPer-species annotation counts:")
    for concept in species_list:
        cnt = species_counts[concept]
        bar = "#" * min(cnt, 50)
        print(f"  {concept:40s}: {cnt:4d} {bar}")

    # Phase 2: Download images
    print(f"\nDownloading {len(all_images)} images...")

    # Map URL to local path
    url_to_path = {}
    for i, (url, meta) in enumerate(all_images.items()):
        ext = url.split(".")[-1].split("?")[0]
        if ext not in ("png", "jpg", "jpeg"):
            ext = "png"
        url_to_path[url] = images_dir / f"{i:06d}.{ext}"

    semaphore = asyncio.Semaphore(args.concurrent)

    async def download_limited(client, url, path):
        async with semaphore:
            return await download_image(client, url, path)

    downloaded = 0
    failed_urls = set()
    async with httpx.AsyncClient(http2=False) as client:
        tasks = []
        urls_list = list(all_images.keys())
        for url in urls_list:
            tasks.append(download_limited(client, url, url_to_path[url]))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, result in zip(urls_list, results):
            if isinstance(result, Exception) or result is False:
                failed_urls.add(url)
            else:
                downloaded += 1

    print(f"  Downloaded: {downloaded}, Failed: {len(failed_urls)}")

    # Phase 3: Crop ROIs at multiple scales
    print(f"\nCropping ROIs at scales {args.scales}...")

    coco_images = []
    coco_annotations = []
    csv_rows = []
    img_id_counter = 0
    ann_id_counter = 0

    # Build category map
    category_map = {name: i + 1 for i, name in enumerate(species_list)}

    # Track which images we've added to COCO
    url_to_img_id = {}

    successful = 0
    for ann in tqdm(annotations, desc="Cropping ROIs"):
        img_url = ann["image_url"]
        if img_url in failed_urls:
            continue

        img_path = url_to_path[img_url]
        if not img_path.exists():
            continue

        img_meta = all_images[img_url]
        img_w = img_meta["width"]
        img_h = img_meta["height"]

        # If width/height not in API metadata, read from file
        if img_w == 0 or img_h == 0:
            try:
                with Image.open(img_path) as pil_img:
                    img_w, img_h = pil_img.size
                    img_meta["width"] = img_w
                    img_meta["height"] = img_h
            except Exception:
                continue

        bbox = ann["bbox"]

        # Skip invalid bboxes
        if bbox[2] < 2 or bbox[3] < 2:
            continue

        # Assign COCO image ID
        if img_url not in url_to_img_id:
            img_id_counter += 1
            url_to_img_id[img_url] = img_id_counter
            coco_images.append({
                "id": img_id_counter,
                "width": img_w,
                "height": img_h,
                "file_name": img_path.name,
                "coco_url": img_url,
            })

        img_id = url_to_img_id[img_url]
        ann_id_counter += 1

        # Crop at each scale
        all_crops_ok = True
        for scale in args.scales:
            if scale == 1.0:
                scaled_bbox = [int(b) for b in bbox]
            else:
                scaled_bbox = expand_bbox(bbox, scale, img_w, img_h)

            scale_name = f"{int(scale)}x" if scale == int(scale) else f"{scale}x"
            roi_dir = rois_dir / scale_name
            roi_path = roi_dir / f"{img_id}_{ann_id_counter}.png"

            if not crop_roi(img_path, scaled_bbox, roi_path):
                all_crops_ok = False

        # Also link full image (hard link is instant vs slow re-encode)
        full_dir = rois_dir / "full"
        full_path = full_dir / f"{img_id}_{ann_id_counter}.png"
        if not full_path.exists():
            try:
                full_dir.mkdir(parents=True, exist_ok=True)
                os.link(img_path, full_path)
            except OSError:
                # Fallback to copy if hard link fails (cross-device)
                import shutil
                try:
                    shutil.copy2(img_path, full_path)
                except Exception:
                    all_crops_ok = False

        if all_crops_ok:
            successful += 1
            concept = ann["concept"]
            cat_id = category_map[concept]

            coco_annotations.append({
                "id": ann_id_counter,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "segmentation": [],
                "iscrowd": 0,
            })

            roi_1x = rois_dir / "1x" / f"{img_id}_{ann_id_counter}.png"
            csv_rows.append([str(roi_1x.resolve()), concept])

    print(f"  Successfully cropped: {successful}/{len(annotations)} annotations")

    # Phase 4: Save COCO JSON and CSV
    coco_categories = [{"id": i + 1, "name": name} for i, name in enumerate(species_list)]

    coco_dataset = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    coco_path = output_dir / "dataset_external.json"
    with open(coco_path, "w") as f:
        json.dump(coco_dataset, f, indent=2)
    print(f"\nSaved COCO JSON: {coco_path} ({len(coco_annotations)} annotations)")

    # Save annotations CSV
    csv_path = output_dir / "annotations.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(csv_rows)
    print(f"Saved annotations CSV: {csv_path}")

    # Save solution CSV (for evaluation with evaluate_on_test.py)
    solution_path = output_dir / "solution_external.csv"
    with open(solution_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["annotation_id", "concept_name"])
        for ann in coco_annotations:
            concept = species_list[ann["category_id"] - 1]
            writer.writerow([ann["id"], concept])
    print(f"Saved solution CSV: {solution_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"External Validation Dataset Summary")
    print(f"{'='*70}")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"  Total images: {len(coco_images)}")
    print(f"  Total annotations: {len(coco_annotations)}")
    print(f"  Species covered: {sum(1 for s in species_list if species_counts[s] > 0)}/{len(species_list)}")
    print(f"  ROI scales: {args.scales} + full")
    print(f"\n  Per-species breakdown:")

    final_counts = defaultdict(int)
    for ann in coco_annotations:
        final_counts[ann["category_id"]] += 1
    for i, name in enumerate(species_list):
        cnt = final_counts.get(i + 1, 0)
        print(f"    {name:40s}: {cnt:4d}")

    print(f"\n  Files:")
    print(f"    COCO JSON:    {coco_path}")
    print(f"    Annotations:  {csv_path}")
    print(f"    Solution:     {solution_path}")
    print(f"    ROI dirs:     {rois_dir}/{{1x,3x,5x,full}}")


if __name__ == "__main__":
    asyncio.run(main())
