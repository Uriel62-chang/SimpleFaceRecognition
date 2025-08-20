# 人脸识别项目构建脚本 - OpenCV + OpenVINO 版本
# 移除 dlib 依赖，使用更稳定的 OpenCV DNN 模块

param(
    [string]$Config = "Release",
    [string]$BuildType = "default"
)

Write-Host "=== 人脸识别项目构建脚本 ===" -ForegroundColor Green
Write-Host "配置: $Config" -ForegroundColor Yellow
Write-Host "构建类型: $BuildType" -ForegroundColor Yellow

# 检查 CMake 是否可用
try {
    $cmakeVersion = cmake --version 2>&1
    Write-Host "CMake 版本: $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "错误: 未找到 CMake，请先安装 CMake" -ForegroundColor Red
    exit 1
}

# 检查 OpenCV 是否可用
try {
    $opencvVersion = python -c "import cv2; print(cv2.__version__)" 2>&1
    Write-Host "OpenCV 版本: $opencvVersion" -ForegroundColor Green
} catch {
    Write-Host "警告: 未找到 Python OpenCV，但 C++ 版本可能可用" -ForegroundColor Yellow
}

# 创建构建目录
if (!(Test-Path "build")) {
    New-Item -ItemType Directory -Name "build" | Out-Null
    Write-Host "创建构建目录: build/" -ForegroundColor Green
}

# 进入构建目录
Set-Location "build"

# 配置项目
Write-Host "配置项目..." -ForegroundColor Green
cmake .. -DCMAKE_BUILD_TYPE=$Config

if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: CMake 配置失败" -ForegroundColor Red
    exit 1
}

# 构建项目
Write-Host "构建项目..." -ForegroundColor Green
cmake --build . --config $Config

if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 构建失败" -ForegroundColor Red
    exit 1
}

# 检查输出文件
$exePath = "bin\face_recognition.exe"
if (Test-Path $exePath) {
    Write-Host "构建成功！可执行文件: $exePath" -ForegroundColor Green
    
    # 显示文件信息
    $fileInfo = Get-Item $exePath
    Write-Host "文件大小: $($fileInfo.Length / 1MB) MB" -ForegroundColor Cyan
    Write-Host "创建时间: $($fileInfo.CreationTime)" -ForegroundColor Cyan
} else {
    Write-Host "警告: 未找到可执行文件" -ForegroundColor Yellow
}

# 返回项目根目录
Set-Location ".."

Write-Host "=== 构建完成 ===" -ForegroundColor Green
Write-Host "提示: 运行 'cd build\bin\Release && .\face_recognition.exe' 来测试程序" -ForegroundColor Cyan
