# DIP Lab · 数字图像处理本地实验平台

一个在本机启动、在浏览器交互的 Python 图像处理实验台。界面参考用户提供的《课件网页.html》的深色实验台、分模块参数区和原图/处理结果对照形式；算法由 `app.py` 中的 Python/OpenCV 实现，代码可直接修改。

## 实验内容

- 几何变换：最近邻/双线性缩放、任意角旋转（自动扩展画布）、平移、镜像翻转。
- 灰度与单通道：灰度、RGB、HSV、Lab 通道查看。
- 噪声：高斯噪声、椒盐噪声；随机种子可复现。
- 滤波与锐化：均值、高斯、中值、双边、拉普拉斯锐化。
- 边缘检测：Roberts、Prewitt、Sobel、Laplacian、Canny。
- 增强与分割：直方图均衡、CLAHE、全局阈值、Otsu、腐蚀、膨胀、开/闭运算。
- 扩展应用：齿轮/零件检索，使用 Hu 矩、轮廓形状、HOG 和 HSV 直方图进行相似度排序。

项目自带一张由代码绘制的风景示意图，用作流程演示，不是实拍照片。可在左侧上传自己的风景照片；滤波、通道、几何变换和边缘检测均可直接应用到真实照片。项目还附带几何零件示例图库。

## 运行

安装 Anaconda 后，打开 Anaconda Prompt，切换到本目录：

```bat
conda env create -f environment.yml
conda activate dip-lab
streamlit run app.py
```

浏览器会打开本机实验页面。VS Code 中用 `Ctrl+Shift+P` → `Python: Select Interpreter` 选择 `dip-lab`，然后在终端执行同一条 `streamlit run app.py`。

也可使用 pip：

```bat
conda create -n dip-lab python=3.10 -y
conda activate dip-lab
pip install -r requirements.txt
streamlit run app.py
```

## 修改代码

所有主要图像处理函数在 `app.py` 顶部：`transform`、`channel_view`、`add_noise`、`filtering`、`edges`、`basic_enhance`。参数控件和算法模块位于 `processing_page`。修改保存后，Streamlit 通常会自动刷新页面。

## 更新零件图库

把图片按 `dataset/类别名称/图片文件` 放置，选择“齿轮 / 零件相似检索”页面即可使用。每个子目录名称会被当作类别名。图库适合用同类不同角度/外观的图片演示传统特征检索；演示图库是程序生成的简单几何图，不表示真实工业识别精度。

## Windows EXE

此项目采用浏览器本地服务交互，推荐提交 Python 源码及环境文件并使用 `streamlit run app.py` 启动。若课程要求 exe，可在已激活环境中安装 PyInstaller 并尝试：

```bat
pip install pyinstaller
pyinstaller --noconfirm --clean --onedir --name DIP-Lab app.py
```

Streamlit 依赖静态资源和运行时文件，常需额外配置 PyInstaller 的数据文件与隐藏导入；打包结果应在目标 Windows 机器上验证。对本项目而言，提交环境配置和源码是更可靠、也便于老师修改的运行方式。
