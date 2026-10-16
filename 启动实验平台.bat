@echo off
chcp 65001 >nul
python -m streamlit run app.py
if errorlevel 1 (
  echo 启动失败。请先在 Anaconda Prompt 激活 dip-lab 环境并安装 requirements.txt。
  pause
)
