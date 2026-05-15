#!/usr/bin/env python3
"""
启动脚本 - 自动安装依赖并运行应用
"""

import subprocess
import sys
import os

def main():
    print("=" * 50)
    print("🌲 句法分析可视化工具 - 启动脚本")
    print("=" * 50)
    
    # 读取依赖列表
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    # 安装依赖
    print("\n📦 正在安装依赖包...")
    print(f"   需要安装: {', '.join(requirements)}")
    
    for req in requirements:
        package_name = req.split(">=")[0].split("==")[0]
        try:
            __import__(package_name)
            print(f"   ✓ {package_name} 已安装")
        except ImportError:
            print(f"   ⏳ 正在安装 {package_name}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", req],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"   ✓ {package_name} 安装成功")
            else:
                print(f"   ✗ {package_name} 安装失败: {result.stderr}")
    
    # 安装 spacy 模型
    print("\n📥 正在下载语言模型...")
    models = [
        ("spacy", "en_core_web_sm"),
        ("benepar", "benepar_en3")
    ]
    
    for model_type, model_name in models:
        print(f"   ⏳ 正在下载 {model_type} 模型: {model_name}...")
        try:
            if model_type == "spacy":
                result = subprocess.run(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "benepar.download", model_name],
                    capture_output=True,
                    text=True
                )
            
            if result.returncode == 0:
                print(f"   ✓ {model_name} 下载成功")
            else:
                print(f"   ⚠ {model_name} 下载遇到问题，程序启动时会自动重试")
        except Exception as e:
            print(f"   ⚠ 下载出错: {e}")
    
    print("\n" + "=" * 50)
    print("🚀 正在启动应用...")
    print("=" * 50)
    print("应用启动后，请访问: http://localhost:8501")
    print("按 Ctrl+C 停止服务\n")
    
    # 启动 Streamlit 应用
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"])

if __name__ == "__main__":
    main()
