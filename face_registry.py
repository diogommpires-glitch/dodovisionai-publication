"""
DODO VISION AI — Face Registry (Folder-Based + API-Compatible)

Provides both:
  - CLI batch registration from named folders (python face_registry.py --scan)
  - Module-level functions for main.py to import and call directly

FOLDER USAGE:
  faces/registry/John_Smith/photo1.jpg
  faces/registry/Maria_Costa/selfie.png
  Then: python face_registry.py --scan

MODULE USAGE (from main.py):
  import face_registry
  face_registry.list_faces()                   -> dict of registered faces
  face_registry.register_face(cv2_image, name) -> bool
  face_registry.delete_face(name)              -> bool
  face_registry.recognize(cv2_image)           -> (name, similarity) or (None, 0)
  face_registry.identify(embedding_list)       -> (name, similarity) or (None, 0)
  face_registry.get_embedding(image_or_path)   -> list[float] or None
"""

import os
import json
import time
import argparse
import numpy as np

# Suppress TF noise when imported as module
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REGISTRY_DIR = os.path.join("faces", "registry")
REGISTRY_FILE = os.path.join("faces", "registry.json")
PHOTO_DIR = os.path.join("faces", "photos")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIMILARITY_THRESHOLD = 0.42


# ══════════════════════════════════════════════════════════════
#  CORE STORAGE
# ══════════════════════════════════════════════════════════════
def _load():
    """Load registry from disk. Returns dict."""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save(data):
    """Save registry to disk."""
    os.makedirs(os.path.dirname(REGISTRY_FILE) or "faces", exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════
#  EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════
def get_embedding(image_or_path, for_registration=False):
    """Extract Facenet512 embedding from an image.
    Accepts: file path (str), numpy array (cv2 image), or bytes.
    for_registration=True uses retinaface detector for better quality.
    Returns: list of 512 floats, or None on failure."""
    from deepface import DeepFace
    import cv2

    img = None

    # Handle different input types
    if isinstance(image_or_path, str):
        if os.path.isfile(image_or_path):
            img = cv2.imread(image_or_path)
        else:
            return None
    elif isinstance(image_or_path, np.ndarray):
        img = image_or_path
    elif isinstance(image_or_path, (bytes, bytearray)):
        arr = np.frombuffer(image_or_path, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        return None

    if img is None or img.size == 0:
        return None

    # Preprocessing: normalize lighting for consistent embeddings
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_ch = clahe.apply(l_ch)
        img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    backend = "retinaface" if for_registration else "skip"

    try:
        result = DeepFace.represent(
            img_path=img,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend=backend,
        )
        if result and len(result) > 0:
            return result[0]["embedding"]
    except Exception:
        pass

    # Fallback: if retinaface failed, try with skip (no detection, uses full image)
    if for_registration:
        try:
            result = DeepFace.represent(
                img_path=img,
                model_name="Facenet512",
                enforce_detection=False,
                detector_backend="skip",
            )
            if result and len(result) > 0:
                return result[0]["embedding"]
        except Exception:
            pass

    return None


def _cosine_sim(a, b):
    """Cosine similarity between two embedding vectors."""
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)


# ══════════════════════════════════════════════════════════════
#  PUBLIC API  (what main.py calls)
# ══════════════════════════════════════════════════════════════
def list_faces():
    """Return dict of all registered faces.
    Format: {name: {"embedding": [...], "photo_count": N, ...}, ...}
    Called by main.py at startup to print count."""
    return _load()


_MAX_SAMPLES_PER_PERSON = 10  # v9.7.5: keep up to N distinct embeddings


def register_face(image, name):
    """Register a face from a cv2 image (numpy array) with a given name.
    Returns True on success, False on failure.

    v9.7.5: switched from single averaged embedding to a multi-sample bag
    (up to `_MAX_SAMPLES_PER_PERSON` per person). The previous averaging
    approach made re-registrations gradually blur the embedding into
    "you across all conditions", which then matched poorly against any
    specific condition (with vs without glasses, different lighting,
    beard grown / shaved). The bag approach keeps each capture as its
    own sample, and `identify()` uses best-of-bag matching — so a clean
    capture under one condition only needs that one sample to match,
    not be diluted by all the other samples.
    """
    if name is None or not str(name).strip():
        return False

    name = str(name).strip()
    emb = get_embedding(image, for_registration=True)
    if emb is None:
        return False

    reg = _load()

    if name in reg:
        # Migrate legacy single-embedding entry to the new multi-sample
        # schema if needed.
        entry = reg[name]
        samples = entry.get("samples")
        if not samples:
            old = entry.get("embedding")
            samples = [old] if old else []
        # De-duplicate near-identical samples (cosine sim > 0.98) so a
        # finger-on-LEARN-FACE rapid-fire doesn't bloat the bag with
        # near-copies of the same frame.
        is_dup = any(_cosine_sim(emb, s) > 0.98 for s in samples)
        if not is_dup:
            samples.append(emb)
        # Cap the bag — keep most recent N (most-recent reflects current
        # appearance better than a 6-month-old capture).
        if len(samples) > _MAX_SAMPLES_PER_PERSON:
            samples = samples[-_MAX_SAMPLES_PER_PERSON:]
        entry["samples"] = samples
        entry["photo_count"] = len(samples)
        entry["embedding"] = np.mean(np.array(samples), axis=0).tolist()  # legacy alias
        entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Save photo if possible
        photo_path = None
        try:
            import cv2
            os.makedirs(PHOTO_DIR, exist_ok=True)
            safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            photo_path = os.path.join(PHOTO_DIR, f"{safe_name}.jpg")
            cv2.imwrite(photo_path, image)
        except Exception:
            pass

        reg[name] = {
            "samples": [emb],          # v9.7.5: new multi-sample schema
            "embedding": emb,           # legacy alias for backward compat
            "photo_count": 1,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_folder": "api",
            "photo": photo_path,
        }

    _save(reg)
    return True


def delete_face(name):
    """Remove a person from the registry by name. Returns True if found."""
    reg = _load()
    # Case-insensitive match
    matches = [k for k in reg if k.lower() == name.lower()]
    if not matches:
        return False
    for m in matches:
        # Remove photo file if exists
        photo = reg[m].get("photo")
        if photo and os.path.isfile(photo):
            try:
                os.remove(photo)
            except OSError:
                pass
        del reg[m]
    _save(reg)
    return True


def recognize(image):
    """Identify a person from a cv2 image (numpy array).
    Returns dict {"name": str|None, "similarity": float} for main.py compatibility."""
    emb = get_embedding(image)
    if emb is None:
        return {"name": None, "similarity": 0}
    name, sim = identify(emb)
    return {"name": name, "similarity": sim}


def identify(embedding):
    """Match an embedding vector against all registered faces.
    Returns (name, similarity) or (None, 0) if below threshold.

    v9.7.5: best-of-bag matching. Compares against EACH sample in the
    person's bag (up to `_MAX_SAMPLES_PER_PERSON`) and uses the highest
    similarity. Far more robust than the old "compare against averaged
    embedding" approach which gave a worst-of-both-worlds match — a
    sample from when you didn't have glasses and one from when you did
    average to a "blurry-glasses" embedding that matched neither
    state well. With the bag approach the without-glasses sample
    matches when you're without glasses, the with-glasses sample
    matches when you are, no dilution.
    """
    if embedding is None:
        return None, 0

    reg = _load()
    best_name = None
    best_sim = 0.0

    for name, data in reg.items():
        # New schema: list of samples. Legacy schema: single embedding
        # stored under either "samples":[one] or "embedding". Try
        # samples first; fall back to embedding.
        samples = data.get("samples")
        if not samples:
            single = data.get("embedding")
            samples = [single] if single is not None else []
        for s in samples:
            if s is None:
                continue
            sim = _cosine_sim(embedding, s)
            if sim > best_sim:
                best_sim = sim
                best_name = name

    if best_sim >= SIMILARITY_THRESHOLD:
        return best_name, round(best_sim, 3)
    return None, 0


def clear():
    """v9.7.5: wipe the entire registry. Returns count of removed
    entries. Used when the user wants to start fresh after appearance
    changes (haircut, beard, new glasses) that even multi-sample
    bags can't bridge."""
    reg = _load()
    n = len(reg)
    _save({})
    # Also clean up the on-disk photo cache
    try:
        import shutil
        if os.path.isdir(PHOTO_DIR):
            shutil.rmtree(PHOTO_DIR)
    except Exception:
        pass
    return n


# Aliases so main.py can call any reasonable name
identify_face = identify
match_face = identify
recognize_face = recognize
remove_face = delete_face
remove_person = delete_face
get_registered = list_faces
load_registry = _load
save_registry = _save


def register(name, image):
    """Register face — main.py calls this as face_registry.register(name, image).
    Note: arg order is (name, image), not (image, name).
    Returns dict response for FastAPI endpoint."""
    ok = register_face(image, name)
    if ok:
        return {"ok": True, "name": name}
    return {"ok": False, "error": "No face detected or bad image"}


def unregister(name):
    """Remove face — main.py calls this as face_registry.unregister(name).
    Returns dict response for FastAPI endpoint."""
    found = delete_face(name)
    if found:
        return {"ok": True, "removed": name}
    return {"ok": False, "error": f"'{name}' not found in registry"}


# ══════════════════════════════════════════════════════════════
#  FOLDER-BASED BATCH SCAN
# ══════════════════════════════════════════════════════════════
def scan_folders():
    """Walk faces/registry/, process every person folder, build embeddings."""
    if not os.path.isdir(REGISTRY_DIR):
        os.makedirs(REGISTRY_DIR, exist_ok=True)
        print(f"\n  Created {REGISTRY_DIR}/")
        print("  Drop named folders with photos inside, then run again.")
        return 0

    person_dirs = []
    for entry in sorted(os.listdir(REGISTRY_DIR)):
        full_path = os.path.join(REGISTRY_DIR, entry)
        if os.path.isdir(full_path):
            person_dirs.append((entry, full_path))

    if not person_dirs:
        print("\n  No person folders found in faces/registry/")
        print("  Create folders named after people with photos inside.")
        return 0

    print(f"\n  Found {len(person_dirs)} person folder(s):\n")
    for name, path in person_dirs:
        images = [f for f in os.listdir(path)
                  if os.path.splitext(f)[1].lower() in SUPPORTED_EXT]
        print(f"    {name}: {len(images)} photo(s)")

    reg = _load()
    added = 0

    for folder_name, folder_path in person_dirs:
        display_name = folder_name.replace("_", " ").replace("-", " ")
        images = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ])

        if not images:
            print(f"\n  [{folder_name}] No images, skipping")
            continue

        existing = reg.get(display_name)
        if existing and existing.get("photo_count", 0) == len(images):
            print(f"\n  [{display_name}] Already up to date ({len(images)} photos)")
            continue

        print(f"\n  [{display_name}] Processing {len(images)} photos...")
        embeddings = []
        ok = 0
        fail = 0

        for idx, img_file in enumerate(images):
            img_path = os.path.join(folder_path, img_file)
            emb = get_embedding(img_path, for_registration=True)
            if emb is not None:
                embeddings.append(emb)
                ok += 1
            else:
                fail += 1

            done = idx + 1
            pct = int(done / len(images) * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r    [{bar}] {done}/{len(images)} ({pct}%)", end="", flush=True)

        print()

        if not embeddings:
            print(f"    [!] No valid faces found for {display_name}")
            continue

        avg = np.mean(embeddings, axis=0).tolist()
        reg[display_name] = {
            "embedding": avg,
            "photo_count": len(images),
            "successful": ok,
            "failed": fail,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_folder": folder_name,
        }
        added += 1
        print(f"    OK: {ok}/{len(images)} embedded, {fail} failed")

    _save(reg)
    print(f"\n  Done: {added} new, {len(reg)} total in registry\n")
    return added


# Alias
scan_and_register = scan_folders


# ══════════════════════════════════════════════════════════════
#  FASTAPI ROUTE ATTACHMENT  (called from main.py if needed)
# ══════════════════════════════════════════════════════════════
def register_routes(app):
    """Attach face registry HTTP endpoints to an existing FastAPI app."""
    from fastapi import UploadFile, File
    import cv2 as _cv2

    @app.get("/faces/list")
    async def _faces_list():
        reg = _load()
        return {
            "count": len(reg),
            "faces": {
                name: {
                    "photo_count": d.get("photo_count", 0),
                    "registered_at": d.get("registered_at"),
                }
                for name, d in reg.items()
            },
        }

    @app.post("/faces/scan")
    async def _faces_scan():
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, scan_folders)
        return {"ok": True, "registered": len(_load())}

    @app.post("/faces/register")
    async def _faces_register(file: UploadFile = File(...), name: str = ""):
        if not name:
            return {"error": "Provide ?name=PersonName"}
        raw = await file.read()
        arr = np.frombuffer(raw, np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Bad image"}
        ok = register_face(img, name)
        return {"ok": ok, "name": name}

    @app.post("/faces/delete")
    async def _faces_delete(name: str = ""):
        if not name:
            return {"error": "Provide ?name=PersonName"}
        ok = delete_face(name)
        return {"ok": ok}


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════
def _print_list():
    reg = _load()
    if not reg:
        print("\n  No faces registered. Run: python face_registry.py --scan\n")
        return
    print(f"\n  Registered faces ({len(reg)}):\n")
    for name, data in sorted(reg.items()):
        photos = data.get("photo_count", "?")
        when = data.get("registered_at", "?")
        folder = data.get("source_folder", "?")
        print(f"    {name:30s}  {photos:>3} photos  ({when})  [{folder}/]")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dodo Vision AI — Face Registry")
    parser.add_argument("--scan", action="store_true",
                        help="Scan faces/registry/ folders and register all")
    parser.add_argument("--list", action="store_true",
                        help="List all registered faces")
    parser.add_argument("--remove", type=str, default="",
                        help="Remove a person by name")
    args = parser.parse_args()

    if args.scan:
        scan_folders()
    elif args.list:
        _print_list()
    elif args.remove:
        found = delete_face(args.remove)
        if found:
            print(f"  Removed: {args.remove}")
        else:
            print(f"  '{args.remove}' not found")
    else:
        print("\n  DODO VISION AI — Face Registry")
        print("  Usage:")
        print("    python face_registry.py --scan     Scan folders, register faces")
        print("    python face_registry.py --list     List registered people")
        print("    python face_registry.py --remove \"John Smith\"")
        print()
        print("  Folder structure:")
        print("    faces/registry/John_Smith/photo1.jpg")
        print("    faces/registry/Maria_Costa/selfie.png")
        print()
