# getface_gpu_final.py
import os
import sys
import cv2
import bz2
import requests
import numpy as np
import concurrent.futures
from tqdm import tqdm
import argparse
import dlib
from deepface import DeepFace
import threading

class FaceExtractor:
    def __init__(self, output_root="output", anchor_name="anchor"):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self._download_model())
        
        self.output_dir = os.path.join(output_root, anchor_name)
        self.emotion_dirs = {
            'happy': os.path.join(self.output_dir, 'happy'),
            'sad': os.path.join(self.output_dir, 'sad'),
            'neutral': os.path.join(self.output_dir, 'neutral'),
            'angry': os.path.join(self.output_dir, 'angry'),
            'surprise': os.path.join(self.output_dir, 'surprise')
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        for dir_path in self.emotion_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        self.emotion_limits = 20
        self.emotion_counts = {e:0 for e in self.emotion_dirs.keys()}
        self.lock = threading.Lock()  # 新增线程锁

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

    def _fast_face_detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return self.face_detector(gray, 1)

    def process_frame_batch(self, frames_data):
        results = []
        for frame_idx, frame in frames_data:
            try:
                faces = self._fast_face_detect(frame)
                if faces:
                    main_face = max(faces, key=lambda r: (r.right()-r.left())*(r.bottom()-r.top()))
                    landmarks = self.landmark_predictor(frame, main_face)
                    
                    height, width = frame.shape[:2]
                    face_area = (main_face.right()-main_face.left())*(main_face.bottom()-main_face.top())
                    if face_area < (width*height)*0.002:
                        continue
                        
                    y1 = max(0, main_face.top() - int(main_face.height() * 0.3))
                    y2 = min(height, main_face.bottom() + int(main_face.height() * 0.1))
                    x1 = max(0, main_face.left() - int(main_face.width() * 0.1))
                    x2 = min(width, main_face.right() + int(main_face.width() * 0.1))
                    face_img = frame[y1:y2, x1:x2]
                    
                    try:
                        analysis = DeepFace.analyze(
                            img_path=face_img, 
                            actions=['emotion'],
                            enforce_detection=False,
                            detector_backend='skip',
                            silent=True
                        )
                        emotion = analysis[0]['dominant_emotion']
                        emotion = 'surprise' if emotion in ['fear', 'disgust'] else emotion
                        emotion = emotion if emotion in self.emotion_counts else 'neutral'
                        
                        # 使用线程锁控制计数
                        with self.lock:
                            if self.emotion_counts[emotion] < self.emotion_limits:
                                self.emotion_counts[emotion] += 1
                                results.append((frame_idx, face_img, emotion))
                            else:
                                continue  # 超过上限直接跳过
                    except Exception as e:
                        pass
            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {str(e)}")
        return results

    def process_video(self, video_path, num_workers=8):
        print("\n开始加速处理视频...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        batch_size = 32
        frame_skip = 3
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            batch = []
            
            for frame_idx in tqdm(range(total_frames), desc="帧处理"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_skip == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch.append((frame_idx, frame))
                    
                    if len(batch) >= batch_size:
                        futures.append(executor.submit(self.process_frame_batch, batch.copy()))
                        batch.clear()
            
            if batch:
                futures.append(executor.submit(self.process_frame_batch, batch))
            
            # 确保所有结果按顺序处理
            with tqdm(total=len(futures), desc="结果保存") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    batch_results = future.result()
                    for frame_idx, face_img, emotion in batch_results:
                        output_path = os.path.join(self.emotion_dirs[emotion], f"{frame_idx}.jpg")
                        try:
                            cv2.imwrite(
                                output_path,
                                cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                            )
                        except Exception as e:
                            print(f"保存 {output_path} 失败: {str(e)}")
                    pbar.update(1)
        
        cap.release()
        print("\n最终统计：")
        for emotion, count in self.emotion_counts.items():
            print(f"{emotion}: {count}张")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="输入视频路径（示例：test.mp4）")
    parser.add_argument("--anchor", required=True, help="主播名称（示例：张三）")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数（建议值：8-16）")
    args = parser.parse_args()
    
    extractor = FaceExtractor(anchor_name=args.anchor)
    extractor.process_video(args.video, num_workers=args.workers)
