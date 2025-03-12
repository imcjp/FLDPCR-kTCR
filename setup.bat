@echo off
setlocal

:: Step 1: 检查是否已安装 Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 未安装，请安装 Python。
    echo 请访问 https://www.python.org/downloads/ 下载并安装 Python。
    exit /b 1
)


:: Step 2: 创建虚拟环境

:: 解析参数
set force=0
if "%1"=="-f" (
    set force=1
)

if exist "venv" (
    if %force%==1 (
        echo 强制删除现有虚拟环境...
        rmdir /s /q venv
        if %errorlevel% neq 0 (
            echo 删除虚拟环境失败，请检查错误信息。
            exit /b 1
        )
    ) else (
        echo 虚拟环境 'venv' 已存在。
    )
)

if not exist "venv" (
    echo 正在创建虚拟环境...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo 创建虚拟环境失败，请检查错误信息。
        exit /b 1
    )
    echo 虚拟环境创建成功。
)

:: Step 3: 检查虚拟环境的激活脚本是否存在
if not exist "venv\Scripts\activate" (
    echo 警告：找不到虚拟环境的激活脚本 'venv\Scripts\activate'，请检查虚拟环境是否正确创建。
    exit /b 1
)

:: Step 4: 激活虚拟环境
echo 正在激活虚拟环境...
call venv\Scripts\activate.bat

:: Step 5: 安装 requirements.txt 中的 Python 包
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if exist requirements.txt (
    echo 正在安装 requirements.txt 中的依赖...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 安装依赖失败，请检查错误信息。
        exit /b 1
    )
) else (
    echo 未找到 requirements.txt，跳过安装依赖步骤。
)

echo 安装完成，所有步骤成功执行。

echo 测试streamlit，若streamlit能够正常启动，请按Ctrl+z或Ctrl+c退出即可
streamlit run frontEnd.py

endlocal
