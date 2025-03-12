@echo off
setlocal

:: Step 1: ����Ƿ��Ѱ�װ Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python δ��װ���밲װ Python��
    echo ����� https://www.python.org/downloads/ ���ز���װ Python��
    exit /b 1
)


:: Step 2: �������⻷��

:: ��������
set force=0
if "%1"=="-f" (
    set force=1
)

if exist "venv" (
    if %force%==1 (
        echo ǿ��ɾ���������⻷��...
        rmdir /s /q venv
        if %errorlevel% neq 0 (
            echo ɾ�����⻷��ʧ�ܣ����������Ϣ��
            exit /b 1
        )
    ) else (
        echo ���⻷�� 'venv' �Ѵ��ڡ�
    )
)

if not exist "venv" (
    echo ���ڴ������⻷��...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo �������⻷��ʧ�ܣ����������Ϣ��
        exit /b 1
    )
    echo ���⻷�������ɹ���
)

:: Step 3: ������⻷���ļ���ű��Ƿ����
if not exist "venv\Scripts\activate" (
    echo ���棺�Ҳ������⻷���ļ���ű� 'venv\Scripts\activate'���������⻷���Ƿ���ȷ������
    exit /b 1
)

:: Step 4: �������⻷��
echo ���ڼ������⻷��...
call venv\Scripts\activate.bat

:: Step 5: ��װ requirements.txt �е� Python ��
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if exist requirements.txt (
    echo ���ڰ�װ requirements.txt �е�����...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ��װ����ʧ�ܣ����������Ϣ��
        exit /b 1
    )
) else (
    echo δ�ҵ� requirements.txt��������װ�������衣
)

echo ��װ��ɣ����в���ɹ�ִ�С�

echo ����streamlit����streamlit�ܹ������������밴Ctrl+z��Ctrl+c�˳�����
streamlit run frontEnd.py

endlocal
