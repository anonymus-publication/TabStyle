"""Convert selected echo DICOM cine loops to TabStyle AVI inputs."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_cine(path, fps_override=None):
    import pydicom
    from pydicom.pixels import pixel_array

    ds = pydicom.dcmread(path)
    photo = str(ds.PhotometricInterpretation)
    if photo not in {'RGB', 'YBR_FULL', 'YBR_FULL_422', 'MONOCHROME1', 'MONOCHROME2'}:
        raise ValueError(f'Unsupported photometric interpretation: {photo}')
    frames = pixel_array(ds, raw=False)
    color = int(ds.get('SamplesPerPixel', 1)) == 3
    if frames.ndim == (3 if color else 2):
        frames = frames[None]
    if len(frames) < 2:
        raise ValueError('Expected a cine loop with at least two frames')
    if frames.dtype != np.uint8:
        # One scale for the entire cine avoids per-frame brightness flicker.
        lo, hi = float(frames.min()), float(frames.max())
        frames = ((frames.astype(np.float32) - lo) * (255 / max(hi - lo, 1))).clip(0, 255).astype(np.uint8)
    if photo == 'MONOCHROME1':
        frames = 255 - frames
    if not color:
        frames = np.repeat(frames[..., None], 3, axis=-1)
    fps = fps_override
    if fps is None:
        timing = ds.get('FrameTimeVector')
        if timing is not None:
            values = np.asarray(timing, dtype=float)
            values = values[values > 0]
            if values.size and not np.allclose(values, values[0], rtol=0.01):
                raise ValueError('Variable frame timing: resample externally or explicitly choose --fps')
            if values.size:
                fps = 1000 / float(values.mean())
        if fps is None and float(ds.get('FrameTime', 0)) > 0:
            fps = 1000 / float(ds.FrameTime)
        if fps is None:
            fps = float(ds.get('CineRate', 0)) or float(ds.get('RecommendedDisplayFrameRate', 0))
    if not fps or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Missing valid frame timing; supply --fps')
    return frames, float(fps)


def remove_ecg(frame, bottom_ratio):
    """Black out green/cyan pixels only within the configured bottom band."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([40, 60, 40]), np.array([100, 255, 255]))
    mask[:int(frame.shape[0] * (1 - bottom_ratio))] = 0
    result = frame.copy()
    result[mask > 0] = 0
    return result


def motion_mask(frames, threshold=8):
    previous = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    motion = np.zeros(previous.shape, np.float32)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        motion += np.abs(gray - previous)
        previous = gray
    motion /= len(frames) - 1
    image = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.equalizeHist(image)
    mask = (image > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Fill enclosed holes with a padded flood fill (the corner may be foreground).
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    mask = cv2.bitwise_or(padded, cv2.bitwise_not(flood))[1:-1, 1:-1]
    if not mask.any():
        raise ValueError('Empty motion mask; inspect the cine or use --no-motion-mask')
    return mask


def square_resize(frame, size):
    h, w = frame.shape[:2]
    side = max(h, w)
    frame = cv2.copyMakeBorder(frame, (side-h)//2, side-h-(side-h)//2,
                              (side-w)//2, side-w-(side-w)//2,
                              cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def write_avi(path, frames, fps, mask, size):
    # A temporary file prevents failed conversions from appearing complete.
    temporary = path.with_name(path.stem + '.partial.avi')
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'MJPG'), fps, (size, size))
    try:
        if not writer.isOpened():
            raise RuntimeError('Could not open MJPG video writer')
        for frame in frames:
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            writer.write(cv2.cvtColor(square_resize(frame, size), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(temporary))
    try:
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (size, size):
                raise RuntimeError('Output dimensions do not match')
            count += 1
        if count != len(frames):
            raise RuntimeError(f'Output verification failed: {count}/{len(frames)} frames')
    finally:
        capture.release()
        if temporary.exists() and count != len(frames):
            temporary.unlink()
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--size', type=int, default=800)
    parser.add_argument('--fps', type=float, help='Override playback FPS; does not resample frames')
    parser.add_argument('--remove-ecg', action='store_true')
    parser.add_argument('--ecg-bottom-ratio', type=float, default=0.3)
    parser.add_argument('--no-motion-mask', action='store_true')
    parser.add_argument('--motion-threshold', type=int, default=8)
    args = parser.parse_args(argv)
    if args.size < 2 or args.size % 2 or not 0 < args.ecg_bottom_ratio <= 1:
        parser.error('size must be positive and even; ECG bottom ratio must be in (0, 1]')
    if not 0 <= args.motion_threshold <= 255:
        parser.error('motion threshold must be in [0, 255]')
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        parser.error('fps must be finite and positive')
    paths = sorted(p for p in args.input_dir.rglob('*') if p.suffix.lower() == '.dcm' and not p.name.startswith('.'))
    if not paths:
        parser.error('No .dcm files found in input directory')
    ids = [p.stem for p in paths]
    if len(set(x.casefold() for x in ids)) != len(ids):
        parser.error('Duplicate filenames: assign unique deidentified filenames before processing')
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new or empty output directory to prevent overwriting previous results')
    (out / 'videos').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir()
    (out / 'previews').mkdir()
    settings = vars(args).copy()
    # Do not save source directory paths in shareable outputs.
    settings.pop('input_dir')
    settings.pop('output_dir')
    (out / 'settings.json').write_text(json.dumps(settings, indent=2))
    rows, failures = [], []
    for path in paths:
        try:
            frames, fps = read_cine(path, args.fps)
            if args.remove_ecg:
                frames = np.stack([remove_ecg(f, args.ecg_bottom_ratio) for f in frames])
            mask = (np.full(frames.shape[1:3], 255, np.uint8) if args.no_motion_mask
                    else motion_mask(frames, args.motion_threshold))
            write_avi(out / 'videos' / f'{path.stem}.avi', frames, fps, mask, args.size)
            preview = cv2.bitwise_and(frames[len(frames)//2], frames[len(frames)//2], mask=mask)
            for target, image in [(out / 'masks' / f'{path.stem}.png', mask),
                                  (out / 'previews' / f'{path.stem}.png', cv2.cvtColor(square_resize(preview, args.size), cv2.COLOR_RGB2BGR))]:
                if not cv2.imwrite(str(target), image):
                    raise RuntimeError('Could not write review image')
            rows.append({'video_id': path.stem, 'video_name': path.stem,
                         'frames': len(frames), 'fps': fps})
            print(f'OK: {path.stem}')
        except Exception as error:
            failures.append({'video_id': path.stem, 'error_type': type(error).__name__})
            print(f'FAILED: {path.stem}: {error}')
    with (out / 'manifest.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['video_id', 'video_name', 'frames', 'fps'])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'failures.json').write_text(json.dumps(failures, indent=2))
    print(f'{len(rows)} succeeded; {len(failures)} failed. Review previews and masks before training.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
