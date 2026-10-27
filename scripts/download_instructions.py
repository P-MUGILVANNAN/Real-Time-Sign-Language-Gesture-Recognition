"""
Public Sign Language Dataset Guide and Download Instructions (WLASL & Public Datasets).
"""

import sys

GUIDE_TEXT = """
========================================================================================
           SignVision AI — Public Dataset Access & Compliance Guide (WLASL)
========================================================================================

1. DATASET OVERVIEW: WLASL (Word-Level American Sign Language)
   - Dataset Homepage: https://dxli94.github.io/WLASL/
   - GitHub Repository: https://github.com/dxli94/WLASL
   - License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
   - Publication: "Word-level Deep Sign Language Recognition from Video: A New Large-scale
     Dataset and Methods Comparison" (WACV 2020).

2. ACCESS RESTRICTIONS & COMPLIANCE:
   - Due to video licensing constraints, third parties cannot redistribute raw videos directly.
   - Users must download video index metadata (WLASL_v0.3.json) and run the official WLASL
     download script to fetch video instances from original sources (YouTube, DeafSign, ASL-LEX).

3. STEP-BY-STEP ACQUISITION WORKFLOW:
   Step 3.1: Download WLASL Index JSON:
             Download `WLASL_v0.3.json` into `data/raw/wlasl/`

   Step 3.2: Filter Target Classes:
             Filter for target gesture classes: hello, thank_you, yes, no, please, help, sorry.

   Step 3.3: Extract Landmarks with SignVision AI:
             Run:
             python scripts/extract_landmarks.py --input_dir data/raw/wlasl/videos --label hello

   Step 3.4: Train and Evaluate:
             Run:
             python scripts/train_model.py

4. ALTERNATIVE: CUSTOM WEBCAM DATASET (Recommended for Local Zero-Setup Execution):
   - You can also record your own custom dataset directly inside SignVision AI!
   - Navigate to the "Dataset Management" page in the web app:
     streamlit run app.py
   - Select your target gesture, click Record, perform the gesture in front of your camera,
     and the application will automatically extract and save landmark sequences.
========================================================================================
"""

def main():
    print(GUIDE_TEXT)

if __name__ == "__main__":
    main()
