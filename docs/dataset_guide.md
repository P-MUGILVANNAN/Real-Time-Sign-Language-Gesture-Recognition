# SignVision AI — Dataset Acquisition & Management Guide

## 1. Supported Dataset Paradigms
SignVision AI supports two distinct dataset workflows:
1. **Custom Webcam Dataset (Interactive Collection)**: Collect and annotate your own gestures directly via webcam.
2. **Public Datasets (WLASL, LIBRAS, MS-ASL)**: Convert public video sign language benchmarks into landmark sequences.

---

## 2. Option A: Custom Webcam Dataset Workflow

### 2.1. Recording Gestures via UI
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Navigate to **📁 Dataset Management** -> **🎥 Record Custom Gesture**.
3. Select the target gesture class (e.g., `hello`, `thank_you`, `yes`, `no`, `please`, `help`, `sorry`, `iloveyou`).
4. Enter your signer ID (e.g., `signer_user_1`), session ID, background condition, and speed.
5. Click **🔴 Start Recording Session**. A 3-second countdown will prepare you, followed by a 30-frame capture.
6. The application automatically tracks hand presence, normalizes skeletal coordinates, and writes the `.npy` array into `data/custom/<gesture_class>/`.

### 2.2. Metadata Manifest Format (`data/metadata/dataset_manifest.csv`)
Each recorded sample is cataloged with the following metadata schema:
- `sample_id`: Unique identifier (e.g., `hello_1730000000_a1b2c3`).
- `gesture_class`: Target sign category.
- `signer_id`: Signer identifier for group-disjoint splitting.
- `session_id`: Recording session identifier.
- `background`: Environmental background (e.g., `office_desk`, `plain_wall`).
- `speed`: Performance velocity (`normal`, `slow`, `fast`).
- `timestamp`: ISO timestamp of acquisition.
- `num_frames`: Number of raw frames recorded.
- `detection_rate`: Percentage of frames where hands were successfully localized.
- `filepath`: Relative path to the `.npy` landmark sequence.

---

## 3. Option B: Public Dataset (WLASL) Integration

### 3.1. About WLASL
- **Title**: Word-level American Sign Language (WLASL)
- **Publication**: Li et al., IEEE Winter Conference on Applications of Computer Vision (WACV 2020)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
- **Official Repository**: https://github.com/dxli94/WLASL

### 3.2. Compliance & Download Instructions
WLASL consists of web videos hosted on third-party platforms (YouTube, DeafSign, ASL-LEX). Users must obtain the index JSON and download the videos directly using the official scripts:

1. Download `WLASL_v0.3.json` from the official repository and place it in `data/raw/wlasl/`.
2. Run the official WLASL video download script to retrieve instances for your target vocabulary.
3. Use the SignVision batch video feature extractor:
   ```bash
   python scripts/extract_landmarks.py --input_dir data/raw/wlasl/videos/hello --output_dir data/custom/hello --label hello
   ```
4. Update the manifest and compile processed arrays:
   ```bash
   python scripts/prepare_dataset.py
   ```

---

## 4. Synthetic Benchmark Dataset
For automated CI/CD pipeline verification and testing environments where a physical camera is not attached, SignVision AI provides a deterministic benchmark generator:
```bash
python scripts/prepare_dataset.py --generate_benchmark --samples_per_class 25
```
> [!NOTE]
> Synthetic landmark arrays are generated strictly for software integration testing and mathematical pipeline verification; they are never substituted for empirical experimental results.
