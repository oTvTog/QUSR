import os
import glob
import base64
import io
from PIL import Image
import torch
from torch.utils import data as data
from tqdm import tqdm
import argparse
import requests
import json
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default=None, help='Preset directory (default: ../preset)')
parser.add_argument("--batch_size", type=int, default=1, help='批次大小')
parser.add_argument("--epoch", type=int, default=8, help='生成数据集的轮次')
parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1/chat/completions", help='vLLM服务地址')
parser.add_argument("--vllm_urls", type=str, nargs='+', default=None, help='多个vLLM服务地址列表')
parser.add_argument("--max_retries", type=int, default=3, help='最大重试次数')
parser.add_argument("--wait_for_service", type=int, default=60, help='等待服务启动的最大时间（秒）')
parser.add_argument("--start_gpu", type=int, default=0, help='起始GPU编号（0-3）')
parser.add_argument("--end_gpu", type=int, default=4, help='结束GPU编号（1-4）')
parser.add_argument("--image_size", type=int, default=512, help='图像处理尺寸，减小可提高速度')
parser.add_argument("--max_workers", type=int, default=16, help='并行处理的最大工作线程数')
parser.add_argument("--max_words", type=int, default=80, help='生成提示词的最大单词数')
parser.add_argument("--gpu_id", type=int, default=0, help='当前GPU ID（0-3）')
    parser.add_argument("--model_path", type=str, default="/data2/Solar_Data/PiSA-SR/Qwen2.5-VL-7B-Instruct", help='vLLM模型路径')
    args = parser.parse_args()


def check_vllm_service(base_url, max_wait_time=60):
    """检查vLLM服务是否正常运行"""
    print(f"检查vLLM服务状态: {base_url}")
    
    # 提取基础URL
    base_url = base_url.replace('/v1/chat/completions', '')
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            # 检查模型列表端点
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"✅ vLLM服务正常运行")
                print(f"可用模型: {models}")
                return True
            else:
                print(f"⚠️ 服务响应异常: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"⏳ 等待服务启动... ({int(time.time() - start_time)}s)")
        except Exception as e:
            print(f"⚠️ 检查服务时出错: {e}")
        
        time.sleep(2)
    
    print(f"❌ 服务启动超时 ({max_wait_time}s)")
    return False

@lru_cache(maxsize=1000)
def encode_image_to_base64_cached(image_path):
    """缓存图像编码结果，避免重复编码"""
    try:
        # 读取并预处理图像
        with Image.open(image_path) as img:
            # 转换为RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整图像大小以提高速度
            if max(img.size) > args.image_size:
                img.thumbnail((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            
            # 保存到内存缓冲区
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            
            # 编码为base64
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        print(f"图像编码失败 {image_path}: {e}")
        return None

def encode_image_to_base64(image_path):
    """将图像编码为base64字符串"""
    return encode_image_to_base64_cached(image_path)

def clean_generated_text(text):
    """清理生成的文本，确保格式正确"""
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = text.strip()
    
    # 如果文本以不完整的句子结尾，尝试修复
    if text.endswith(('result in', 'contribute to', 'lead to', 'cause', 'make', 'create')):
        # 移除不完整的句子
        sentences = text.split('.')
        if len(sentences) > 1:
            # 保留完整的句子
            text = '. '.join(sentences[:-1]) + '.'
    
    return text

def call_vllm_api(image_path, prompt, max_retries=3):
    """调用vLLM API进行图像描述生成 - 完全匹配XPSR格式"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # 编码图像为base64（使用缓存）
    base64_image = encode_image_to_base64(image_path)
    if base64_image is None:
        return ""
    
    # 准备请求数据
    payload = {
        "model": args.model_path,  
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 100,  # 减少token数，配合60词限制
        "temperature": 0.1,  # 降低随机性，更稳定
        "top_p": 0.6,  # 更聚焦的生成        
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(args.vllm_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                # 获取原始响应并清理
                raw_response = result['choices'][0]['message']['content']
                cleaned_response = clean_generated_text(raw_response)
                return cleaned_response
            else:
                print(f"API响应格式异常: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"API调用失败: {image_path}")
                return ""
        except Exception as e:
            print(f"处理响应时出错: {str(e)}")
            return ""

# 定义数据集类 - 直接处理LR图像
class LocalImageDataset(data.Dataset):
    def __init__(self, pngtxt_dir, image_size=512, start=0, end=1):
        super(LocalImageDataset, self).__init__()
        self.lr_img_paths = []  # 直接存储LR图像路径
        self.image_size = image_size

        # 直接从预生成的LR图像目录读取路径
        lr_dir = os.path.join(pngtxt_dir, 'pre_generated_lr')
        if os.path.exists(lr_dir):
            # 扫描LR目录中的所有PNG文件
            lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.png')))
            print(f"从 {lr_dir} 目录读取到 {len(lr_paths)} 张LR图像")
            
            # 数据集分割逻辑
            if lr_paths and end > start:
                total_images = len(lr_paths)
                # 计算每个GPU处理的数据量
                num_gpus = end - start  # 实际使用的GPU数量
                images_per_gpu = total_images // num_gpus
                remainder = total_images % num_gpus  # 余数
                
                start_idx = start * images_per_gpu
                # 将余数分配给前面的GPU
                if start < remainder:
                    start_idx += start
                    end_idx = start_idx + images_per_gpu + 1
                else:
                    start_idx += remainder
                    end_idx = start_idx + images_per_gpu
                
                # 确保不超过总图像数
                if end_idx > total_images:
                    end_idx = total_images
                
                self.lr_img_paths = lr_paths[start_idx:end_idx]
                print(f"GPU {start}-{end-1}: 处理范围 [{start_idx}:{end_idx}]，共 {len(self.lr_img_paths)} 张LR图像")
            else:
                print(f"错误: start_gpu ({start}) 必须小于 end_gpu ({end}) 或没有找到LR图像")
                self.lr_img_paths = []
        else:
            print(f"警告: 未找到 {lr_dir} 目录")
            # 回退到从gt_path.txt推断LR路径
            gt_path_file = os.path.join(pngtxt_dir, 'gt_path.txt')
            if os.path.exists(gt_path_file):
                with open(gt_path_file, 'r', encoding='utf-8') as f:
                    hr_paths = [line.strip() for line in f.readlines() if line.strip()]
                
                # 将HR路径转换为LR路径
                lr_paths = []
                for hr_path in hr_paths:
                    if '/HR/' in hr_path:
                        lr_path = hr_path.replace("/HR/", "/LR/")
                        # 检查LR文件是否存在
                        if os.path.exists(lr_path):
                            lr_paths.append(lr_path)
                        else:
                            print(f"警告: LR文件不存在 {lr_path}")
                    else:
                        # 如果路径格式不标准，尝试其他方式
                        lr_paths.append(hr_path)  # 暂时使用原路径
                
                print(f"从 {gt_path_file} 推断出 {len(lr_paths)} 条LR路径")
                
                # 应用相同的分割逻辑
                if lr_paths and end > start:
                    total_images = len(lr_paths)
                    num_gpus = end - start
                    images_per_gpu = total_images // num_gpus
                    remainder = total_images % num_gpus
                    
                    start_idx = start * images_per_gpu
                    if start < remainder:
                        start_idx += start
                        end_idx = start_idx + images_per_gpu + 1
                    else:
                        start_idx += remainder
                        end_idx = start_idx + images_per_gpu
                    
                    if end_idx > total_images:
                        end_idx = total_images
                    
                    self.lr_img_paths = lr_paths[start_idx:end_idx]
                    print(f"GPU {start}-{end-1}: 处理范围 [{start_idx}:{end_idx}]，共 {len(self.lr_img_paths)} 张LR图像")
                else:
                    self.lr_img_paths = []
            else:
                print(f"警告: 未找到 {gt_path_file} 文件")
                self.lr_img_paths = []

    def __getitem__(self, index):
        # 直接返回LR图像路径
        lr_path = self.lr_img_paths[index]
        return lr_path

    def __len__(self):
        return len(self.lr_img_paths)

# 定义LR图像质量评估的提示 - 简洁且完整
lr_text_prompt = 'In 60 words or less, describe this low-resolution image, evaluating its quality based on clarity, color, noise, and lighting.'

# HR图像提示已移除 - 只处理LR图像

# 检查vLLM服务状态
if not check_vllm_service(args.vllm_url, args.wait_for_service):
    print("❌ vLLM服务未正常运行，请先启动服务")
    print("启动命令示例:")
    print("python start_vllm_service.py --model_path /data2/Solar_Data/PiSA-SR/Qwen2.5-VL-7B-Instruct --port 8000")
    exit(1)

# 初始化数据集和 DataLoader
train_dataset = LocalImageDataset(
    pngtxt_dir=args.save_dir, 
    start=args.start_gpu, 
    end=args.end_gpu
)
train_dataset_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,  
    num_workers=8, 
    pin_memory=True,
    shuffle=False
)

def process_single_image(args_tuple):
    """处理单张图像（用于并行处理）"""
    image_path, save_path, prompt = args_tuple
    
    try:
        # 调用vLLM API生成描述
        response = call_vllm_api(image_path, prompt, args.max_retries)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存生成的描述
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.replace('\n', ' '))
        
        return True, len(response), save_path
    except Exception as e:
        print(f"处理图像失败 {image_path}: {str(e)}")
        # 保存空文件
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('')
        return False, 0, save_path

# 处理图像并生成描述 - 使用并行处理
print("开始使用vLLM进行图像质量评估...")
print(f"vLLM服务地址: {args.vllm_url}")
print(f"并行工作线程数: {args.max_workers}")

# 统计信息
description_stats = {
    'total_processed': 0,
    'successful': 0,
    'failed': 0,
    'lengths': [],
    'avg_length': 0
}

# 准备所有任务
all_tasks = []

with torch.no_grad():
    for i, example in enumerate(train_dataset_loader):
        # 直接获取LR图像路径
        lr_path = example[0]  # 解包batch
        
        # 构建输出文件路径 - 只处理LR图像
        if '/LR/' in lr_path:
            # 标准XPSR格式
            lr_save_path = lr_path.replace("/LR/", "/lowlevel_prompt_q/").replace(".png", ".txt")
        else:
            # 从文件读取的路径格式
            base_dir = args.save_dir
            lr_filename = os.path.basename(lr_path).replace('.png', '.txt')
            
            lr_output_dir = os.path.join(base_dir, 'lowlevel_prompt_q')
            os.makedirs(lr_output_dir, exist_ok=True)
            
            lr_save_path = os.path.join(lr_output_dir, lr_filename)

        # 跳过已存在的LR输出文件
        if os.path.exists(lr_save_path):
            continue

        # 只添加LR图像处理任务（不处理HR图像）
        all_tasks.append((lr_path, lr_save_path, lr_text_prompt))

# 使用ThreadPoolExecutor进行并行处理
print(f"开始并行处理 {len(all_tasks)} 个任务...")
start_time = time.time()

with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    # 提交所有任务
    future_to_task = {
        executor.submit(process_single_image, task): task 
        for task in all_tasks
    }
    
    # 使用tqdm显示进度
    with tqdm(total=len(all_tasks), desc="处理进度") as pbar:
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            image_path, save_path, prompt = task
            
            try:
                success, length, save_path = future.result()
                description_stats['total_processed'] += 1
                
                if success:
                    description_stats['successful'] += 1
                    description_stats['lengths'].append(length)
                else:
                    description_stats['failed'] += 1
                    # 记录异常
                    with open(os.path.dirname(save_path) + "_abnormal.txt", 'a') as f:
                        f.write(os.path.basename(save_path) + "\n")
                
            except Exception as e:
                print(f"任务执行失败: {str(e)}")
                description_stats['failed'] += 1
            
            pbar.update(1)

# 计算处理时间
end_time = time.time()
total_time = end_time - start_time
avg_time_per_image = total_time / len(all_tasks) if all_tasks else 0

# 计算统计信息
if description_stats['lengths']:
    description_stats['avg_length'] = sum(description_stats['lengths']) / len(description_stats['lengths'])

print("\n" + "="*50)
print("LR图像提示词生成统计信息:")
print(f"总处理LR图像: {description_stats['total_processed']}")
print(f"成功生成: {description_stats['successful']}")
print(f"生成失败: {description_stats['failed']}")
print(f"平均提示词长度: {description_stats['avg_length']:.1f} 字符")
if description_stats['lengths']:
    print(f"提示词长度范围: {min(description_stats['lengths'])} - {max(description_stats['lengths'])} 字符")
print(f"总处理时间: {total_time:.2f} 秒")
print(f"平均每张LR图像处理时间: {avg_time_per_image:.2f} 秒")
print(f"处理速度: {len(all_tasks)/total_time:.2f} LR图像/秒")
print("="*50)

print("LR图像质量评估完成！")