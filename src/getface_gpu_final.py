# getface_gpu_final.py
import os
import sys
import time
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
import threading
import argparse

# 检查依赖库
try:
    import dlib
    from deepface import DeepFace
except ImportError as e:
    print(f"缺少依赖库: {str(e)}")
    sys.exit(1)

# GPU配置检查
if dlib.DLIB_USE_CUDA:
    print("检测到CUDA加速支持")
else:
    print("警告：未启用CUDA加速，将使用CPU模式")

class FaceExtractor:
    def __init__(self, output_dir="output"):
        # 初始化模型路径
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self._download_model())
        self.output_dir = output_dir
        self.emotion_counts = {'happy':0, 'sad':0, 'neutral':0, 'angry':0, 'surprise':0}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def _download_model(self):
        """自动下载并返回shape_predictor模型路径"""
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            print("正在下载面部特征点模型...")
            os.system(f"wget https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            os.system(f"bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
        return model_path

    def _check_face_quality(self, frame, face_rect, landmarks):
        """增强版人脸质量检测"""
        quality = {
            "is_good": True,
            "reasons": []
        }

        # 计算人脸区域面积
        face_area = (face_rect.right() - face_rect.left()) * (face_rect.bottom() - face_rect.top())
        img_area = frame.shape[0] * frame.shape[1]
        
        # 面积检查
        if face_area < img_area * 0.002:  # 人脸面积小于图像0.2%
            quality["is_good"] = False
            quality["reasons"].append("face_too_small")

        # 关键点可见性检查
        jaw_points = [landmarks.part(i) for i in range(0, 17)]
        for point in jaw_points:
            if (point.x < 0 or point.x >= frame.shape[1] or 
                point.y < 0 or point.y >= frame.shape[0]):
                quality["is_good"] = False
                quality["reasons"].append("face_out_of_frame")
                break
        return quality

    def process_frame_batch(self, frames_data):
        """处理一批视频帧"""
        results = []
        for frame_idx, frame in frames_data:
            try:
                # 检测人脸
                faces = self.face_detector(frame, 1)
                
                if len(faces) > 0:
                    # 选择最大的人脸
                    main_face = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
                    
                    # 检测面部特征点
                    landmarks = self.landmark_predictor(frame, main_face)
                    
                    # 判断是否是好的人脸
                    face_quality = self._check_face_quality(frame, main_face, landmarks)
                    if face_quality["is_good"]:
                        # 裁剪人脸区域（扩展版）
                        height, width = frame.shape[:2]
                        y1 = max(0, main_face.top() - int(main_face.height() * 0.3))
                        y2 = min(height, main_face.bottom() + int(main_face.height() * 0.1))
                        x1 = max(0, main_face.left() - int(main_face.width() * 0.1))
                        x2 = min(width, main_face.right() + int(main_face.width() * 0.1))
                        
                        face_img = frame[y1:y2, x1:x2]
                        
                        # 增强版表情分析
                        try:
                            analysis = DeepFace.analyze(
                                img_path=face_img, 
                                actions=['emotion'], 
                                enforce_detection=False,
                                detector_backend='skip',
                                silent=True
                            )
                            
                            if isinstance(analysis, list) and len(analysis) > 0:
                                emotion = analysis[0]['dominant_emotion']
                                
                                # 表情分类映射
                                if emotion in ['fear', 'disgust']:
                                    emotion = 'surprise'
                                elif emotion not in ['happy', 'sad', 'neutral', 'angry', 'surprise']:
                                    emotion = 'neutral'
                                
                                # 保存计数控制
                                if self.emotion_counts[emotion] < 20:
                                    self.emotion_counts[emotion] += 1
                                    results.append((frame_idx, face_img, emotion))
                        except Exception as e:
                            pass
            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {str(e)}")
        return results

    def process_video(self, video_path, num_workers=4):
        """多线程处理视频并提取人脸"""
        print("\n开始处理视频...")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 多线程处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            batch = []
            
            for frame_idx in tqdm(range(total_frames), desc="处理进度"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % 5 != 0:  # 间隔采样
                    continue
                
                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append((frame_idx, frame))
                
                # 批量提交任务
                if len(batch) >= 8:
                    futures.append(executor.submit(self.process_frame_batch, batch))
                    batch = []
            
            # 处理剩余批次
            if len(batch) > 0:
                futures.append(executor.submit(self.process_frame_batch, batch))
            
            # 保存结果
            for future in tqdm(concurrent.futures.as_completed(futures), desc="保存结果"):
                batch_results = future.result()
                for frame_idx, face_img, emotion in batch_results:
                    emotion_dir = os.path.join(self.output_dir, emotion)
                    os.makedirs(emotion_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(emotion_dir, f"frame_{frame_idx}.jpg"), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        
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
