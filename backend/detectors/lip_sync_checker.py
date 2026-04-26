import cv2
import numpy as np
import mediapipe as mp
import librosa
import subprocess
import os
import tempfile
import typing
import logging
import glob
import shutil


logger = logging.getLogger(__name__)


def check_lip_sync(video_path: str) -> dict:
    tmp_dir = ""
    try:
        logger.debug(f"Lip sync check: {video_path}")

        tmp_dir = tempfile.mkdtemp()

        frames_pattern = os.path.join(tmp_dir, "frame_%04d.png")
        audio_path = os.path.join(tmp_dir, "audio.wav")

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-t",
                "5",
                "-vf",
                "fps=25",
                frames_pattern,
                "-y",
                "-loglevel",
                "error",
            ],
            check=True,
        )

        frames = sorted(glob.glob(tmp_dir + "/frame_*.png"))
        if not frames:
            raise RuntimeError("ffmpeg failed to extract frames")

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-t",
                "5",
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                audio_path,
                "-y",
                "-loglevel",
                "error",
            ],
            check=True,
        )

        if not os.path.exists(audio_path):
            raise RuntimeError("ffmpeg failed to extract audio")

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        hop_length = 16000 // 25
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        audio_energy = rms.tolist()

        mp_face_mesh = mp.solutions.face_mesh
        face_detected_count = 0
        mouth_openness_series = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        ) as face_mesh:
            for frame_path in frames:
                img = cv2.imread(frame_path)
                if img is None:
                    mouth_openness_series.append(0.0)
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_detected_count += 1
                    landmarks = results.multi_face_landmarks[0].landmark

                    upper_lip = landmarks[13]
                    lower_lip = landmarks[14]
                    top = landmarks[10]
                    bottom = landmarks[152]

                    face_height = abs(top.y - bottom.y)
                    if face_height > 0:
                        mouth_open = abs(upper_lip.y - lower_lip.y) / face_height
                    else:
                        mouth_open = 0.0
                    mouth_openness_series.append(mouth_open)
                else:
                    mouth_openness_series.append(0.0)

        min_len = min(len(audio_energy), len(mouth_openness_series))
        audio_energy = audio_energy[:min_len]
        mouth_series = mouth_openness_series[:min_len]

        if min_len < 5 or face_detected_count == 0:
            return {
                "sync_score": 0.5,
                "correlation": 0.0,
                "flags": ["no_face_for_sync"],
                "frames_analyzed": len(frames),
                "method": "mediapipe+librosa",
            }

        audio_arr = np.array(audio_energy, dtype=np.float32)
        mouth_arr = np.array(mouth_series, dtype=np.float32)

        if np.std(audio_arr) < 1e-8 or np.std(mouth_arr) < 1e-8:
            correlation = 0.0
        else:
            corr_matrix = np.corrcoef(audio_arr, mouth_arr)
            correlation = float(corr_matrix[0, 1])
            if np.isnan(correlation):
                correlation = 0.0

        sync_score = max(0.0, min(1.0, correlation / 0.3))

        flags = []
        if correlation < 0.12:
            flags.append("poor_lip_sync")
        if face_detected_count == 0:
            flags.append("no_face_for_sync")

        return {
            "sync_score": round(sync_score, 4),
            "correlation": round(correlation, 4),
            "flags": flags,
            "frames_analyzed": min_len,
            "method": "mediapipe+librosa",
        }
    except Exception as e:
        return {
            "sync_score": 0.5,
            "flags": ["sync_check_failed"],
            "correlation": 0.0,
            "frames_analyzed": 0,
            "method": "mediapipe+librosa",
            "error": str(e),
        }
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
