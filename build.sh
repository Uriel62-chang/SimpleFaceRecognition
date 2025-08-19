#!/bin/bash

# 人脸识别项目构建脚本 - OpenCV + OpenVINO 版本 (Linux/macOS)
# 移除 dlib 依赖，使用更稳定的 OpenCV DNN 模块

# 默认配置
CONFIG=${1:-"Release"}
BUILD_TYPE=${2:-"default"}

echo "=== 人脸识别项目构建脚本 ==="
echo "配置: $CONFIG"
echo "构建类型: $BUILD_TYPE"

# 检查 CMake 是否可用
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 CMake，请先安装 CMake"
    echo "Ubuntu/Debian: sudo apt-get install cmake"
    echo "CentOS/RHEL: sudo yum install cmake"
    echo "macOS: brew install cmake"
    exit 1
fi

# 显示 CMake 版本
CMAKE_VERSION=$(cmake --version | head -n1)
echo "CMake 版本: $CMAKE_VERSION"

# 检查 OpenCV 是否可用 (可选检查)
if command -v python3 &> /dev/null; then
    if python3 -c "import cv2; print('OpenCV 版本:', cv2.__version__)" 2>/dev/null; then
        echo "Python OpenCV 可用"
    else
        echo "警告: 未找到 Python OpenCV，但 C++ 版本可能可用"
    fi
else
    echo "警告: 未找到 Python3，跳过 OpenCV 版本检查"
fi

# 创建构建目录
if [ ! -d "build" ]; then
    mkdir -p build
    echo "创建构建目录: build/"
else
    echo "构建目录已存在: build/"
fi

# 进入构建目录
cd build

# 配置项目
echo "配置项目..."
cmake .. -DCMAKE_BUILD_TYPE=$CONFIG

if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败"
    cd ..
    exit 1
fi

# 构建项目
echo "构建项目..."
cmake --build . --config $CONFIG

if [ $? -ne 0 ]; then
    echo "错误: 构建失败"
    cd ..
    exit 1
fi

# 检查输出文件
EXE_PATH="bin/face_recognition"
if [ -f "$EXE_PATH" ]; then
    echo "构建成功！可执行文件: $EXE_PATH"
    
    # 显示文件信息
    FILE_SIZE=$(du -h "$EXE_PATH" | cut -f1)
    FILE_PERM=$(ls -l "$EXE_PATH" | awk '{print $1}')
    echo "文件大小: $FILE_SIZE"
    echo "文件权限: $FILE_PERM"
    
    # 尝试设置执行权限
    chmod +x "$EXE_PATH" 2>/dev/null
    echo "已设置执行权限"
else
    echo "警告: 未找到可执行文件"
fi

# 返回项目根目录
cd ..

echo "=== 构建完成 ==="
echo "提示: 运行 'cd build/bin && ./face_recognition' 来测试程序"

# 显示构建结果摘要
if [ -f "build/bin/face_recognition" ]; then
    echo ""
    echo "🎉 构建成功！"
    echo "📁 可执行文件位置: build/bin/face_recognition"
    echo "🚀 运行命令: ./build/bin/face_recognition"
else
    echo ""
    echo "❌ 构建可能失败，请检查上面的错误信息"
fi
