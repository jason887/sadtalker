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
import time
from deepface import DeepFace
import threading
import tensorflow as tf
import shutil
import uuid
from PIL import Image  # 添加PIL库来处理中文路径保存
import platform
import tempfile

# 设置更友好的错误输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TF警告

# 尝试配置GPU加速
def setup_gpu():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # 设置GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # 检查是否真正可以访问GPU
            with tf.device('/GPU:0'):
                # 简单的GPU测试
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            
            print(f"✓ GPU已启用: 检测到{len(gpus)}个GPU设备")
            return True
        else:
            print("✗ 未检测到GPU设备，将使用CPU模式")
            return False
    except Exception as e:
        print(f"✗ GPU初始化失败 ({str(e)}), 将使用CPU模式")
        return False

# 修复OpenCV模型路径
def fix_opencv_model_path():
    try:
        import pkg_resources
        import os
        
        # 检查DeepFace使用的模型路径
        deepface_path = os.path.dirname(DeepFace.__file__)
        data_folder = os.path.join(deepface_path, "detectors", "opencv")
        cascade_file = os.path.join(data_folder, "haarcascade_frontalface_default.xml")
        
        # 如果路径不存在，尝试从OpenCV复制
        if not os.path.exists(cascade_file):
            print("修复OpenCV人脸检测器路径...")
            os.makedirs(data_folder, exist_ok=True)
            
            # 从OpenCV包获取原始文件位置
            try:
                opencv_path = os.path.dirname(cv2.__file__)
                original_file = os.path.join(opencv_path, "data", "haarcascade_frontalface_default.xml")
                
                if os.path.exists(original_file):
                    shutil.copy(original_file, cascade_file)
                    print(f"✓ 模型文件已复制到: {cascade_file}")
                else:
                    try:
                        # 尝试第二种方法
                        opencv_data = pkg_resources.resource_filename('cv2', 'data')
                        original_file = os.path.join(opencv_data, 'haarcascade_frontalface_default.xml')
                        if os.path.exists(original_file):
                            shutil.copy(original_file, cascade_file)
                            print(f"✓ 模型文件已复制到: {cascade_file}")
                        else:
                            print("✗ 无法找到OpenCV人脸检测器文件")
                    except:
                        print("✗ 无法加载OpenCV资源")
            except Exception as e:
                print(f"修复OpenCV路径失败: {str(e)}")
        
        return True
    except Exception as e:
        print(f"✗ 尝试修复模型路径时出错: {str(e)}")
        return False

class FaceExtractor:
    def __init__(self, output_root="output", anchor_name="anchor", use_ascii=True):
        print("\n初始化人脸提取器...")
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(self._download_model())
        self.use_ascii = use_ascii  # 是否使用纯ASCII文件名
        self.system_encoding = sys.getfilesystemencoding()
        
        print(f"系统文件编码: {self.system_encoding}")
        print(f"操作系统: {platform.system()} {platform.release()}")
        
        # 处理输出路径
        if self.use_ascii:
            # 使用纯ASCII路径
            self.session_id = str(uuid.uuid4())[:8]
            self.output_dir = os.path.join(output_root, f"faces_{self.session_id}")
            print(f"使用ASCII文件名模式，会话ID: {self.session_id}")
            self.anchor_name = anchor_name  # 仅记录但不用于路径
        else:
            # 尝试使用原有路径（可能包含中文）
            self.output_dir = os.path.join(output_root, "faces_" + anchor_name)
            self.anchor_name = anchor_name
        
        print(f"输出目录: {self.output_dir}")
        
        # 先清理旧目录，确保结果干净
        if os.path.exists(self.output_dir):
            try:
                shutil.rmtree(self.output_dir)
                print(f"清理旧目录: {self.output_dir}")
            except Exception as e:
                print(f"清理旧目录失败: {str(e)}")
                # 尝试重命名目录
                backup_dir = self.output_dir + "_bak_" + str(int(time.time()))
                try:
                    os.rename(self.output_dir, backup_dir)
                    print(f"已将旧目录重命名为: {backup_dir}")
                except:
                    print("无法重命名旧目录，将尝试继续")
        
        # 确保根目录存在
        try:
            os.makedirs(output_root, exist_ok=True)
        except Exception as e:
            print(f"创建输出根目录失败: {str(e)}")
            # 尝试使用临时目录
            output_root = tempfile.mkdtemp()
            print(f"将使用临时目录: {output_root}")
            self.output_dir = os.path.join(output_root, f"faces_{self.session_id if self.use_ascii else anchor_name}")
        
        # 创建主目录
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"创建目录: {self.output_dir}")
            # 测试目录是否可写
            test_file = os.path.join(self.output_dir, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print("✓ 输出目录可写入")
        except Exception as e:
            print(f"创建/测试主目录失败: {str(e)}")
            # 使用临时目录
            self.output_dir = tempfile.mkdtemp()
            print(f"将使用临时目录: {self.output_dir}")
        
        # 创建表情目录
        self.emotion_dirs = {
            'happy': os.path.join(self.output_dir, 'happy'),
            'sad': os.path.join(self.output_dir, 'sad'),
            'neutral': os.path.join(self.output_dir, 'neutral'),
            'angry': os.path.join(self.output_dir, 'angry'),
            'surprise': os.path.join(self.output_dir, 'surprise')
        }
        
        # 创建表情分类目录
        for emotion, dir_path in self.emotion_dirs.items():
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"创建目录: {dir_path}")
                # 测试目录是否可写
                test_file = os.path.join(dir_path, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                print(f"创建/测试表情目录失败 ({emotion}): {str(e)}")
        
        # 每类表情最多保存的数量
        self.emotion_limits = 20
        self.emotion_counts = {e: 0 for e in self.emotion_dirs.keys()}
        self.lock = threading.Lock()  # 线程锁
        self.last_save_time = {}  # 用于控制保存频率
        
        # 尝试修复DeepFace的OpenCV路径
        fix_opencv_model_path()
        
        # 预热DeepFace
        try:
            print("预热DeepFace分析引擎...")
            test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _ = DeepFace.analyze(test_img, actions=['emotion'], enforce_detection=False, silent=True)
            print("✓ DeepFace预热成功")
        except Exception as e:
            print(f"✗ DeepFace预热失败: {str(e)}")
            print("  首次处理可能较慢")

    def _download_model(self):
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            print("正在下载面部特征点模型...")
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            temp_file = "shape_predictor_68_face_landmarks.dat.bz2"
            
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(temp_file, 'wb') as f, tqdm(
                    desc="下载进度", total=total_size, unit='iB',
                    unit_scale=True, unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)

                print("解压模型文件...")
                with bz2.open(temp_file, 'rb') as source:
                    with open(model_path, 'wb') as target:
                        target.write(source.read())
                os.remove(temp_file)
                print(f"✓ 模型文件已保存至: {model_path}")
                
            except Exception as e:
                print(f"✗ 模型下载失败: {str(e)}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                sys.exit(1)
        return model_path

    def _check_face_quality(self, frame, face, landmarks):
        """评估人脸质量"""
        result = {"is_good": False, "clarity": 0, "angle_ratio": 0, "size_ratio": 0}
        
        try:
            # 检查人脸大小
            face_width = face.right() - face.left()
            img_width = frame.shape[1]
            size_ratio = face_width / img_width
            result["size_ratio"] = size_ratio
            
            if size_ratio < 0.05:  # 进一步降低尺寸要求
                return result
                
            # 检查面部朝向
            left_eye = np.mean(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]), axis=0)
            right_eye = np.mean(np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]), axis=0)
            nose = np.array([landmarks.part(33).x, landmarks.part(33).y])
            
            eye_center = (left_eye + right_eye) / 2
            angle_ratio = abs(nose[0] - eye_center[0]) / face_width
            result["angle_ratio"] = angle_ratio
            
            if angle_ratio > 0.3:  # 再放宽角度限制
                return result
                
            # 检查图像清晰度
            face_img = frame[face.top():face.bottom(), face.left():face.right()]
            if face_img.size == 0:
                return result
                
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            clarity = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            result["clarity"] = clarity
            
            if clarity < 8:  # 降低清晰度要求
                return result
                
            # 所有检查通过
            result["is_good"] = True
            return result
            
        except Exception as e:
            return result

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
                        # 裁剪人脸区域，增加上下文
                        height, width = frame.shape[:2]
                        y1 = max(0, main_face.top() - int(main_face.height() * 0.3))
                        y2 = min(height, main_face.bottom() + int(main_face.height() * 0.1))
                        x1 = max(0, main_face.left() - int(main_face.width() * 0.1))
                        x2 = min(width, main_face.right() + int(main_face.width() * 0.1))
                        
                        # 确保边界有效
                        if y2 <= y1 or x2 <= x1:
                            continue
                        
                        face_img = frame[y1:y2, x1:x2].copy()
                        
                        # 确保图像有效
                        if face_img.size == 0 or face_img is None:
                            continue
                        
                        # 分析表情
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
                                
                                # 简化表情分类
                                if emotion in ['fear', 'disgust']:
                                    emotion = 'surprise'
                                elif emotion not in ['happy', 'sad', 'neutral', 'angry', 'surprise']:
                                    emotion = 'neutral'
                                
                                # 将结果添加到列表
                                results.append((frame_idx, face_img.copy(), emotion))
                        except:
                            pass
            except:
                pass
                
        return results

    def save_results(self, batch_results):
        """安全地保存处理结果，使用PIL替代OpenCV处理中文路径问题"""
        current_time = time.time()
        saved_count = 0
        
        with self.lock:  # 确保线程安全
            for frame_idx, face_img, emotion in batch_results:
                # 检查是否已达到该表情类别的上限
                if self.emotion_counts[emotion] >= self.emotion_limits:
                    continue
                    
                # 避免保存太相似的帧（同一表情至少间隔0.5秒）
                if emotion in self.last_save_time and current_time - self.last_save_time[emotion] < 0.5:
                    continue
                
                # 保存图像
                try:
                    # 生成唯一的文件名（带随机UUID）
                    unique_id = str(uuid.uuid4())[:8]
                    if self.use_ascii:
                        # 纯ASCII文件名
                        filename = f"{emotion}_{self.emotion_counts[emotion]:03d}_{unique_id}.jpg"
                    else:
                        filename = f"{emotion}_{self.emotion_counts[emotion]:03d}.jpg"
                    
                    save_path = os.path.join(self.emotion_dirs[emotion], filename)
                    
                    # 确保图像格式正确 - 从RGB转为BGR (OpenCV)或保持RGB (PIL)
                    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                        save_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    else:
                        save_img_rgb = face_img
                    
                    # 确保图像有效
                    if save_img_rgb.size == 0 or save_img_rgb is None:
                        continue
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    
                    # 方法1：使用PIL保存（更好地支持中文路径）
                    success = False
                    try:
                        pil_img = Image.fromarray(save_img_rgb)
                        pil_img.save(save_path)
                        success = True
                    except Exception as e:
                        print(f"PIL保存失败: {str(e)}")
                        
                        try:
                            # 方法2：使用OpenCV保存
                            save_img_bgr = cv2.cvtColor(save_img_rgb, cv2.COLOR_RGB2BGR)
                            success = cv2.imwrite(save_path, save_img_bgr)
                            if not success:
                                print(f"OpenCV保存失败: {save_path}")
                        except Exception as e:
                            print(f"OpenCV保存出错: {str(e)}")
                            
                            try:
                                # 方法3：尝试使用临时文件
                                temp_dir = tempfile.gettempdir()
                                temp_path = os.path.join(temp_dir, f"face_{unique_id}.jpg")
                                cv2.imwrite(temp_path, cv2.cvtColor(save_img_rgb, cv2.COLOR_RGB2BGR))
                                shutil.copy(temp_path, save_path)
                                os.remove(temp_path)
                                success = True
                            except:
                                success = False
                        
                    if success:
                        # 更新计数和时间戳
                        self.emotion_counts[emotion] += 1
                        self.last_save_time[emotion] = current_time
                        saved_count += 1
                        
                        if saved_count % 5 == 0:  # 减少日志输出频率
                            print(f"已保存 {saved_count} 个表情图像")
                except Exception as e:
                    print(f"保存图像时出错: {type(e).__name__} - {str(e)}")
        
        return saved_count

    def process_video(self, video_path, num_workers=4):
        """处理视频并提取人脸表情"""
        print("\n开始加速处理视频...")
        
        # 开始计时
        start_time = time.time()
        
        # 检查视频文件
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在 - {video_path}")
            return False
            
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 - {video_path}")
            return False
            
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"视频信息: 总帧数={total_frames}, FPS={fps:.2f}, 时长={duration:.2f}秒")
        print(f"使用 {num_workers} 个线程并行处理")
        
        # 处理参数
        batch_size = 4  # 更小的批次提高稳定性
        frame_skip = 3  # 更小间隔增加检测率
        
        # 创建进度条
        pbar = tqdm(total=total_frames // frame_skip, desc="帧处理")
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            frame_idx = 0
            batch = []
            
            while frame_idx < total_frames:
                # 设置读取位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    # 读取失败时尝试重试
                    cap.release()
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"视频重新打开失败，跳过剩余帧")
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"重试读取帧 {frame_idx} 失败，跳到下一帧")
                        frame_idx += frame_skip
                        pbar.update(1)
                        continue
                
                # 添加到批处理
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB
                batch.append((frame_idx, frame))
                
                # 达到批大小时提交处理
                if len(batch) >= batch_size:
                    futures.append(executor.submit(self.process_frame_batch, batch.copy()))
                    batch.clear()
                    
                # 更新进度并处理下一帧
                pbar.update(1)
                frame_idx += frame_skip
                
                # 检查是否所有表情类别都已达到上限
                with self.lock:
                    all_full = all(count >= self.emotion_limits for count in self.emotion_counts.values())
                    if all_full:
                        print("\n已收集到足够的表情样本，提前结束处理")
                        break
            
            # 处理剩余的批次
            if batch:
                futures.append(executor.submit(self.process_frame_batch, batch))
            
            # 收集处理结果
            total_saved = 0
            results_pbar = tqdm(total=len(futures), desc="结果保存")
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    saved = self.save_results(batch_results)
                    total_saved += saved
                    results_pbar.update(1)
                except Exception as e:
                    print(f"处理结果时出错: {str(e)}")
            
            results_pbar.close()
        
        # 关闭进度条和视频
        pbar.close()
        cap.release()
        
        # 显示处理统计
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n处理完成! 各表情样本数量:")
        for emotion, count in self.emotion_counts.items():
            print(f"{emotion}: {count}张")
        
        print(f"\n总计保存了 {total_saved} 张图像")
        print(f"处理用时: {processing_time:.2f}秒")
        
        # 验证所有目录有文件
        empty_dirs = []
        for emotion, directory in self.emotion_dirs.items():
            files = os.listdir(directory)
            if not files:
                empty_dirs.append(emotion)
                
        if empty_dirs:
            print(f"\n警告: 以下表情类别没有保存任何图像: {', '.join(empty_dirs)}")
            
        # 最终输出结果存储位置
        if total_saved > 0:
            print(f"\n所有表情图像已保存到: {os.path.abspath(self.output_dir)}")
            if self.use_ascii != (self.anchor_name == "anchor"):
                print(f"注意：使用了ASCII文件名模式，原始锚点名称 '{self.anchor_name}' 未用于文件路径")
                
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从视频提取人脸表情')
    parser.add_argument('--video', required=True, help='视频文件路径')
    parser.add_argument('--anchor', default='anchor', help='锚点名称（输出文件夹名）')
    parser.add_argument('--workers', type=int, default=4, help='并行线程数')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU模式')
    parser.add_argument('--ascii', action='store_true', help='使用纯ASCII文件名(解决中文路径问题)')
    args = parser.parse_args()
    
    # 如果指定了CPU模式，禁用GPU
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("已强制使用CPU模式")
    else:
        # 尝试设置GPU
        gpu_available = setup_gpu()
        if not gpu_available:
            print("将继续使用CPU进行处理")
    
    # 创建并运行人脸提取器 - 默认启用ASCII文件名解决中文路径问题
    extractor = FaceExtractor(anchor_name=args.anchor, use_ascii=True)
    extractor.process_video(args.video, num_workers=args.workers)
