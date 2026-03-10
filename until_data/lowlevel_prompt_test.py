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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUSR_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_PRESET = os.path.normpath(os.path.join(_QUSR_ROOT, '..', 'preset'))

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default=None, help='Preset directory (default: ../preset relative to QUSR)')
parser.add_argument("--test_lr_dir", type=str, default=None, help='测试集LR图像目录 (默认: preset/test_datasets/RealSR_test/test_SR_bicubic)')
parser.add_argument("--output_dir", type=str, default=None, help='提示词输出目录 (默认: preset/test_lowlevel_prompt_q_RealSR)')
parser.add_argument("--batch_size", type=int, default=1, help='批次大小')
parser.add_argument("--epoch", type=int, default=8, help='生成数据集的轮次')
parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1/chat/completions", help='vLLM服务地址')
parser.add_argument("--max_retries", type=int, default=3, help='最大重试次数')
parser.add_argument("--wait_for_service", type=int, default=60, help='等待服务启动的最大时间（秒）')
parser.add_argument("--image_size", type=int, default=512, help='图像处理尺寸')
parser.add_argument("--max_workers", type=int, default=8, help='并行处理的最大工作线程数')
parser.add_argument("--max_words", type=int, default=60, help='生成提示词的最大单词数')
args = parser.parse_args()

# 设置默认路径 (QUSR 相对父项目 preset)
if args.test_lr_dir is None:
    args.test_lr_dir = os.path.join(_PRESET, 'test_datasets', 'RealSR_test', 'test_SR_bicubic')
if args.output_dir is None:
    args.output_dir = os.path.join(_PRESET, 'test_lowlevel_prompt_q_RealSR')


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
                print(f"⏳ 等待服务启动... ({int(time.time() - start_time)}s)")
        except requests.exceptions.RequestException:
            print(f"⏳ 等待服务启动... ({int(time.time() - start_time)}s)")
        
        time.sleep(2)
    
    print(f"❌ 服务启动超时 ({max_wait_time}s)")
    print("❌ vLLM服务未正常运行，请先启动服务")
    print("启动命令示例:")
    print("python start_vllm_service.py --model_path /data2/Solar_Data/PiSA-SR/Qwen2.5-VL-7B-Instruct --port 8000")
    return False


def encode_image_to_base64(image_path):
    """将图像编码为base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"编码图像失败 {image_path}: {e}")
        return None


def clean_generated_text(text):
    """清理生成的文本"""
    if not text:
        return ""
    text = text.strip()
    if text.endswith(('result in', 'contribute to', 'lead to', 'cause', 'make', 'create')):
        sentences = text.split('.')
        if len(sentences) > 1:
            text = '. '.join(sentences[:-1]) + '.'
    return text


def call_vllm_api(image_path, prompt, vllm_url, max_retries=3):
    """调用vLLM API生成图像描述"""
    for attempt in range(max_retries):
        try:
            # 编码图像
            base64_image = encode_image_to_base64(image_path)
            if not base64_image:
                return None
            
            # 构建请求数据
            data = {
                "model": "/data2/Solar_Data/PiSA-SR/Qwen2.5-VL-7B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.6
            }
            
            # 发送请求
            response = requests.post(vllm_url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content'].strip()
                return clean_generated_text(generated_text)
            else:
                print(f"API请求失败 (尝试 {attempt + 1}/{max_retries}): {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                
        except Exception as e:
            print(f"调用API时出错 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return None


class TestImageDataset(data.Dataset):
    def __init__(self, test_lr_dir=None, output_dir=None):
        super(TestImageDataset, self).__init__()
        test_lr_dir = test_lr_dir or args.test_lr_dir
        output_dir = output_dir or args.output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有LR图像路径
        self.lr_paths = sorted(glob.glob(os.path.join(test_lr_dir, '*.png')))
        self.output_dir = output_dir
        
        print(f"找到 {len(self.lr_paths)} 张测试集LR图像")

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        # 生成对应的提示词文件路径
        img_name = os.path.basename(lr_path)
        prompt_filename = img_name.replace('.png', '.txt')
        prompt_path = os.path.join(self.output_dir, prompt_filename)
        
        return lr_path, prompt_path

    def __len__(self):
        return len(self.lr_paths)


# 提示词模板
lr_text_prompt = 'In 60 words or less, describe this low-resolution image, evaluating its quality based on clarity, color, noise, and lighting.'


def process_single_image(lr_path, prompt_path, vllm_url):
    """处理单张图像"""
    try:
        # 检查提示词文件是否已存在
        if os.path.exists(prompt_path):
            return f"跳过: {os.path.basename(prompt_path)} (已存在)"
        
        # 调用vLLM API生成提示词
        generated_text = call_vllm_api(lr_path, lr_text_prompt, vllm_url)
        
        if generated_text:
            # 保存提示词文件
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            return f"成功: {os.path.basename(prompt_path)}"
        else:
            return f"失败: {os.path.basename(prompt_path)} (API调用失败)"
            
    except Exception as e:
        return f"错误: {os.path.basename(prompt_path)} - {str(e)}"


def main():
    # 检查vLLM服务
    if not check_vllm_service(args.vllm_url, args.wait_for_service):
        return

    # 创建测试数据集 (使用命令行指定的路径)
    test_dataset = TestImageDataset(test_lr_dir=args.test_lr_dir, output_dir=args.output_dir)
    test_dataset_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=False
    )
    
    print(f"开始使用vLLM进行测试集图像质量评估...")
    print(f"vLLM服务地址: {args.vllm_url}")
    print(f"并行工作线程数: {args.max_workers}")
    
    # 收集所有任务
    tasks = []
    for i, (lr_path, prompt_path) in enumerate(test_dataset_loader):
        lr_path = lr_path[0]
        prompt_path = prompt_path[0]
        tasks.append((lr_path, prompt_path))
    
    print(f"开始并行处理 {len(tasks)} 个任务...")
    
    # 并行处理
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_image, lr_path, prompt_path, args.vllm_url): (lr_path, prompt_path)
            for lr_path, prompt_path in tasks
        }
        
        # 收集结果
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="处理进度"):
            result = future.result()
            results.append(result)
            if "成功" in result:
                print(f"✅ {result}")
            elif "跳过" in result:
                print(f"⏭️ {result}")
            else:
                print(f"❌ {result}")
    
    # 统计信息
    success_count = sum(1 for r in results if "成功" in r)
    skip_count = sum(1 for r in results if "跳过" in r)
    fail_count = sum(1 for r in results if "失败" in r or "错误" in r)
    
    print("\n" + "="*50)
    print("测试集LR图像提示词生成统计信息:")
    print(f"总处理LR图像: {len(tasks)}")
    print(f"成功生成: {success_count}")
    print(f"跳过(已存在): {skip_count}")
    print(f"生成失败: {fail_count}")
    print("="*50)
    print("测试集图像质量评估完成！")


if __name__ == "__main__":
    main()
