# getface_gpu_final.py
import os
import sys
import time
import bz2
import requests
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
import argparse

# 检查依赖库
try:
    import dlib
    from deepface import DeepFace
except ImportError as e:
    print(f"缺少依赖库: {str(e)}")
    sys.exit(1)

class FaceExtractor:
    def __init__(self, output_dir="output"):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self._download_model())
        self.output_dir = output_dir
        self.emotion_counts = {'happy':0, 'sad':0, 'neutral':0, 'angry':0, 'surprise':0}
        os.makedirs(output_dir, exist_ok=True)

    def _download_model(self):
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            print("正在下载面部特征点模型...")
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            temp_file = "shape_predictor_68_face_landmarks.dat.bz2"
            
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(temp_file, 'wb') as f, tqdm(
                    desc="下载进度", total=total_size, unit='iB',
                    unit_scale=True, unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)

                with bz2.open(temp_file, 'rb') as source:
                    with open(model_path, 'wb') as target:
                        target.write(source.read())
                os.remove(temp_file)
                
            except Exception as e:
                print(f"模型下载失败: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                sys.exit(1)
        return model_path

    def _check_face_quality(self, frame, face_rect, landmarks):
        quality = {"is_good": True, "reasons": []}
        face_area = (face_rect.right() - face_rect.left()) * (face_rect.bottom() - face_rect.top())
        img_area = frame.shape[0] * frame.shape[1]
        
        if face_area < img_area * 0.002:
            quality["is_good"] = False
            quality["reasons"].append("face_too_small")

        jaw_points = [landmarks.part(i) for i in range(0, 17)]
        for point in jaw_points:
            if (point.x < 0 or point.x >= frame.shape[1] or 
                point.y < 0 or point.y >= frame.shape[0]):
                quality["is_good"] = False
                quality["reasons"].append("face_out_of_frame")
                break
        return quality

    def process_frame_batch(self, frames_data):
        results = []
        for frame_idx, frame in frames_data:
            try:
                faces = self.face_detector(frame, 1)
                if len(faces) > 0:
                    main_face = max(faces, key=lambda rect: (rect.right()-rect.left())*(rect.bottom()-rect.top()))
                    landmarks = self.landmark_predictor(frame, main_face)
                    
                    if self._check_face_quality(frame, main_face, landmarks)["is_good"]:
                        height, width = frame.shape[:2]
                        y1 = max(0, main_face.top() - int(main_face.height() * 0.3))
                        y2 = min(height, main_face.bottom() + int(main_face.height() * 0.1))
                        x1 = max(0, main_face.left() - int(main_face.width() * 0.1))
                        x2 = min(width, main_face.right() + int(main_face.width() * 0.1))  # 修复的括号
                        
                        face_img = frame[y1:y2, x1:x2]
                        
                        try:
                            analysis = DeepFace.analyze(
                                img_path=face_img, actions=['emotion'],
                                enforce_detection=False, detector_backend='skip', silent=True
                            )
                            if isinstance(analysis, list) and len(analysis) > 0:
                                emotion = analysis[0]['dominant_emotion']
                                emotion = 'surprise' if emotion in ['fear', 'disgust'] else \
                                         ('neutral' if emotion not in self.emotion_counts else emotion)
                                
                                if self.emotion_counts[emotion] < 20:
                                    self.emotion_counts[emotion] += 1
                                    results.append((frame_idx, face_img, emotion))
                        except Exception as e:
                            pass
            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {str(e)}")
        return results

    def process_video(self, video_path, num_workers=4):
        print("\n开始处理视频...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            batch = []
            
            for frame_idx in tqdm(range(total_frames), desc="处理进度"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % 5 == 0:  # 间隔采样
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch.append((frame_idx, frame))
                    
                    if len(batch) >= 8:
                        futures.append(executor.submit(self.process_frame_batch, batch.copy()))
                        batch.clear()
            
            if batch:
                futures.append(executor.submit(self.process_frame_batch, batch))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="保存结果"):
                for frame_idx, face_img, emotion in future.result():
                    emotion_dir = os.path.join(self.output_dir, emotion)
                    os.makedirs(emotion_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(emotion_dir, f"frame_{frame_idx}.jpg"), 
                               cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        
        cap.release()
        print("\n处理完成！各表情样本数量：")
        for emotion, count in self.emotion_counts.items():
            print(f"{emotion}: {count}张")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--workers", type=int, default=4, help="并行工作线程数")
    args = parser.parse_args()
    
    extractor = FaceExtractor()
    extractor.process_video(args.video, num_workers=args.workers)
