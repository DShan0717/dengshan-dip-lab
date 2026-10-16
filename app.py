"""DIP Lab 本地图像处理教学实验平台。

运行：streamlit run app.py
所有图像算法均在本文件中实现，便于课程学习和自行修改。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
FEATURE_CACHE = ROOT / "cache" / "features.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


st.set_page_config(page_title="DIP Lab · 数字图像处理实验平台", page_icon="🔬", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 75% 0%, #14264b 0%, #0b1220 50%); color:#dbe6f7}
[data-testid="stSidebar"] {background:#0e1729}
.hero {padding:1rem 1.2rem;border:1px solid #263858;border-radius:16px;background:linear-gradient(115deg,#14233e,#111b30);margin-bottom:1rem}
.hero h1 {font-size:1.8rem;margin:0;color:#e7efff}.hero p {margin:.4rem 0 0;color:#94a8c9}
[data-testid="stMetric"] {background:#121e34;padding:.75rem;border:1px solid #263858;border-radius:12px}
</style>
""", unsafe_allow_html=True)


def decode_upload(uploaded) -> np.ndarray:
    raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片无法解码，请尝试 PNG、JPG 或 BMP 格式。")
    h, w = image.shape[:2]
    scale = min(1.0, 1400 / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def transform(image: np.ndarray, action: str, params: dict) -> np.ndarray:
    h, w = image.shape[:2]
    if action == "缩放":
        scale = params["scale"]
        interpolation = cv2.INTER_NEAREST if params["interpolation"] == "最近邻" else cv2.INTER_LINEAR
        return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interpolation)
    if action == "旋转":
        angle = params["angle"]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
        out_w, out_h = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
        matrix[0, 2] += out_w / 2 - center[0]
        matrix[1, 2] += out_h / 2 - center[1]
        return cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if action == "平移":
        matrix = np.float32([[1, 0, params["dx"] * w], [0, 1, params["dy"] * h]])
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if action == "镜像翻转":
        return cv2.flip(image, {"左右": 1, "上下": 0, "上下左右": -1}[params["axis"]])
    return image.copy()


def channel_view(image: np.ndarray, space: str, channel: str) -> np.ndarray:
    if space == "RGB":
        channels = cv2.split(image)
        idx = {"蓝 B": 0, "绿 G": 1, "红 R": 2}[channel]
        return channels[idx]
    if space == "灰度":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if space == "HSV":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return converted[{"色相 H": 0, "饱和度 S": 1, "明度 V": 2}[channel]]
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return converted[{"亮度 L": 0, "绿红 a": 1, "蓝黄 b": 2}[channel]]


def add_noise(image: np.ndarray, kind: str, sigma: float, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "高斯噪声":
        noise = rng.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out = image.copy()
    count = int(image.shape[0] * image.shape[1] * density / 2)
    ys = rng.integers(0, image.shape[0], count * 2)
    xs = rng.integers(0, image.shape[1], count * 2)
    out[ys[:count], xs[:count]] = 0
    out[ys[count:], xs[count:]] = 255
    return out


def filtering(image: np.ndarray, kind: str, kernel: int, sigma: float, strength: int) -> np.ndarray:
    if kind == "均值滤波":
        return cv2.blur(image, (kernel, kernel), borderType=cv2.BORDER_REPLICATE)
    if kind == "高斯滤波":
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    if kind == "中值滤波":
        return cv2.medianBlur(image, kernel)
    if kind == "双边滤波（保边）":
        return cv2.bilateralFilter(image, kernel, strength * 2, strength * 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    sharpened = cv2.addWeighted(gray, 1.0, cv2.convertScaleAbs(lap), strength / 10, 0)
    return sharpened


def edges(image: np.ndarray, kind: str, threshold: int, threshold2: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.2)
    if kind == "Canny":
        return cv2.Canny(smooth, threshold, threshold2)
    if kind == "Sobel":
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        return cv2.convertScaleAbs(np.clip(magnitude, 0, 255))
    if kind == "Prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
        ky = kx.T
        gx, gy = cv2.filter2D(smooth, cv2.CV_32F, kx), cv2.filter2D(smooth, cv2.CV_32F, ky)
        return cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    if kind == "Roberts":
        gx = cv2.filter2D(smooth, cv2.CV_32F, np.array([[1, 0], [0, -1]], np.float32))
        gy = cv2.filter2D(smooth, cv2.CV_32F, np.array([[0, 1], [-1, 0]], np.float32))
        return cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    return cv2.convertScaleAbs(cv2.Laplacian(smooth, cv2.CV_32F))


def basic_enhance(image: np.ndarray, method: str, threshold: int, kernel: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if method == "直方图均衡化":
        return cv2.equalizeHist(gray)
    if method == "CLAHE 局部均衡":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if method == "全局阈值分割":
        return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    if method == "Otsu 自动阈值":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    op = {"腐蚀": cv2.MORPH_ERODE, "膨胀": cv2.MORPH_DILATE, "开运算": cv2.MORPH_OPEN, "闭运算": cv2.MORPH_CLOSE}[method]
    return cv2.morphologyEx(binary, op, se)


def feature(image: np.ndarray) -> dict:
    """零件检索描述子：Hu 矩 + 轮廓形状 + HOG + HSV 颜色直方图。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidates = [cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2) for m in (binary, cv2.bitwise_not(binary))]
    candidates = [m if cv2.countNonZero(m) < m.size * .92 else cv2.bitwise_not(m) for m in candidates]
    mask = max(candidates, key=cv2.countNonZero)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    hu = cv2.HuMoments(cv2.moments(contour)).flatten() if contour is not None else np.zeros(7)
    hu = [-math.copysign(math.log10(abs(float(v)) + 1e-30), float(v)) for v in hu]
    area = cv2.contourArea(contour) if contour is not None else 0
    perimeter = cv2.arcLength(contour, True) if contour is not None else 0
    x, y, w, h = cv2.boundingRect(contour) if contour is not None else (0, 0, 1, 1)
    shape = [4 * math.pi * area / (perimeter * perimeter + 1e-8), w / max(h, 1), *hu]
    hog_img = cv2.resize(mask, (64, 64))
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(hog_img).flatten()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [12, 8], [0, 180, 0, 256]).flatten()
    hist /= np.linalg.norm(hist) + 1e-9
    return {"shape": np.asarray(shape, np.float32), "hog": hog, "color": hist}


def feature_distance(a: dict, b: dict) -> float:
    def d(x, y):
        scale = np.maximum(np.maximum(np.abs(x), np.abs(y)), 1.)
        return float(np.linalg.norm((x - y) / scale) / math.sqrt(max(1, x.size)))
    return .45 * d(a["shape"], b["shape"]) + .35 * d(a["hog"], b["hog"]) + .20 * d(a["color"], b["color"])


@st.cache_data(show_spinner=False)
def list_dataset_images() -> tuple:
    if not DATASET.exists():
        return tuple()
    return tuple(str(p) for p in sorted(DATASET.rglob("*")) if p.suffix.lower() in IMAGE_EXTS)


def run_retrieval(query: np.ndarray) -> list:
    matches = []
    qf = feature(query)
    if not FEATURE_CACHE.exists():
        return matches
    try:
        entries = json.loads(FEATURE_CACHE.read_text(encoding="utf-8")).get("entries", [])
    except (OSError, json.JSONDecodeError):
        entries = []
    for entry in entries:
        filename = str(ROOT / entry["path"])
        arr = np.fromfile(filename, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            continue
        cached = {key: np.asarray(value, np.float32) for key, value in entry["feature"].items()}
        matches.append((feature_distance(qf, cached), entry["label"], filename, im))
    return sorted(matches, key=lambda x: x[0])


def extract_gallery_features() -> int:
    entries = []
    for filename in list_dataset_images():
        arr = np.fromfile(filename, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            continue
        entries.append({"path": str(Path(filename).relative_to(ROOT)), "label": Path(filename).parent.name,
                        "feature": {k: v.tolist() for k, v in feature(im).items()}})
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE.write_text(json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(entries)


def sidebar_image() -> np.ndarray | None:
    st.sidebar.markdown("## 🖼️ 输入图像")
    uploaded = st.sidebar.file_uploader("上传风景照片或零件图", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    demo_path = ROOT / "assets" / "demo_landscape.png"
    if uploaded is not None:
        try:
            st.session_state["input_image"] = decode_upload(uploaded)
            st.session_state["input_name"] = uploaded.name
        except Exception as exc:
            st.sidebar.error(str(exc))
    elif "input_image" not in st.session_state and demo_path.exists():
        st.session_state["input_image"] = decode_upload(type("Upload", (), {"getvalue": lambda self: demo_path.read_bytes()})())
        st.session_state["input_name"] = "风景示意图"
    if "input_image" in st.session_state:
        st.sidebar.caption(f"当前图像：{st.session_state.get('input_name', '图像')} · {st.session_state['input_image'].shape[1]}×{st.session_state['input_image'].shape[0]}")
        if st.sidebar.button("恢复内置风景示意图", use_container_width=True):
            st.session_state.pop("input_image", None)
            st.session_state.pop("input_name", None)
            st.rerun()
        return st.session_state["input_image"]
    return None


def main():
    st.markdown('<div class="hero"><h1>🔬 DIP Lab · 数字图像处理实验平台</h1><p>上传一张风景照片，动手调整算法参数，观察图像变换、噪声、滤波、边缘与通道处理结果。</p></div>', unsafe_allow_html=True)
    image = sidebar_image()
    page = st.sidebar.radio("实验模块", ["基础图像处理", "齿轮 / 零件相似检索", "实验原理与操作"], index=0)
    if image is None:
        st.info("请从左侧上传一张照片。也可将 `assets/demo_landscape.png` 作为内置示例使用。")
        return
    if page == "齿轮 / 零件相似检索":
        retrieval_page(image)
        return
    if page == "实验原理与操作":
        st.markdown("""
### 实验流程
1. 左侧选择内置风景示意图，或上传自己的风景照片（也适用于其他图像）。
2. 选择实验模块并调整参数；右侧即时计算处理结果。
3. 对照原图和结果图，观察像素强度、噪声、纹理和边缘如何变化。
4. 下载当前处理结果，或复制修改 `app.py` 中对应算法函数。

### 实现的课件主题
- **几何变换**：缩放（最近邻/双线性）、旋转（自动扩展画布）、平移和镜像。
- **灰度与单通道**：灰度图、RGB 单通道、HSV 与 Lab 单通道。
- **噪声与滤波**：高斯噪声、椒盐噪声、均值/高斯/中值/双边滤波、拉普拉斯锐化。
- **边缘检测**：Roberts、Prewitt、Sobel、Laplacian、Canny。
- **增强与分割**：直方图均衡化、CLAHE、全局阈值、Otsu、腐蚀/膨胀/开闭运算。
- **应用扩展**：Hu 矩、HOG 和 HSV 直方图的零件相似检索。

结果是经典图像处理方法的教学演示；风景照片适合观察变换、滤波和边缘，阈值与形态学效果会随图像内容而变。
""")
        return
    processing_page(image)


def processing_page(image: np.ndarray):
    module = st.sidebar.selectbox("选择算法模块", ["几何变换", "灰度与单通道", "加噪声", "滤波与锐化", "边缘检测", "增强 / 阈值 / 形态学"])
    result = image.copy()
    with st.sidebar:
        st.markdown("---")
        if module == "几何变换":
            action = st.selectbox("操作", ["缩放", "旋转", "平移", "镜像翻转"])
            p = {}
            if action == "缩放":
                p["scale"] = st.slider("缩放比例", .2, 2.5, 1.0, .05)
                p["interpolation"] = st.radio("插值方法", ["最近邻", "双线性"], horizontal=True)
            elif action == "旋转":
                p["angle"] = st.slider("旋转角度（度）", -180, 180, 30)
            elif action == "平移":
                p["dx"] = st.slider("水平位移比例", -1.0, 1.0, .15, .05)
                p["dy"] = st.slider("垂直位移比例", -1.0, 1.0, .1, .05)
            else:
                p["axis"] = st.selectbox("翻转方向", ["左右", "上下", "上下左右"])
            result = transform(image, action, p)
            explanation = "几何变换改变坐标映射；旋转后的空白区域以黑色填充。"
        elif module == "灰度与单通道":
            space = st.selectbox("颜色空间", ["灰度", "RGB", "HSV", "Lab"])
            options = {"灰度": ["灰度图"], "RGB": ["蓝 B", "绿 G", "红 R"], "HSV": ["色相 H", "饱和度 S", "明度 V"], "Lab": ["亮度 L", "绿红 a", "蓝黄 b"]}[space]
            channel = st.selectbox("显示通道", options)
            result = channel_view(image, space, channel)
            explanation = "单通道图以灰度方式显示；不同通道编码了亮度、颜色或色度信息。"
        elif module == "加噪声":
            kind = st.selectbox("噪声类型", ["高斯噪声", "椒盐噪声"])
            sigma = st.slider("高斯标准差 σ", 0, 80, 20) if kind == "高斯噪声" else 20
            density = st.slider("椒盐密度", .001, .15, .03, .001) if kind == "椒盐噪声" else .03
            seed = st.number_input("随机种子（便于复现）", 0, 999999, 42)
            result = add_noise(image, kind, sigma, density, seed)
            explanation = "噪声用于模拟成像干扰。可随后切换到滤波模块观察去噪与细节损失。"
        elif module == "滤波与锐化":
            kind = st.selectbox("算法", ["均值滤波", "高斯滤波", "中值滤波", "双边滤波（保边）", "拉普拉斯锐化"])
            kernel = st.select_slider("邻域大小", options=[3, 5, 7, 9], value=5)
            sigma = st.slider("高斯 σ", .1, 5.0, 1.2, .1)
            strength = st.slider("滤波/锐化强度", 1, 40, 15)
            result = filtering(image, kind, kernel, sigma, strength)
            explanation = "均值和高斯滤波平滑噪声；中值滤波对椒盐噪声有效；双边滤波尽量保边；拉普拉斯用于锐化。"
        elif module == "边缘检测":
            kind = st.selectbox("算子", ["Canny", "Sobel", "Prewitt", "Roberts", "Laplacian"])
            threshold = st.slider("阈值 / Canny 低阈值", 0, 250, 70)
            threshold2 = st.slider("Canny 高阈值", threshold + 1, 255, max(120, threshold + 1)) if kind == "Canny" else 140
            result = edges(image, kind, threshold, threshold2)
            explanation = "边缘响应图突出灰度变化区域；Canny 使用平滑、梯度、非极大值抑制和双阈值连接。"
        else:
            method = st.selectbox("操作", ["直方图均衡化", "CLAHE 局部均衡", "全局阈值分割", "Otsu 自动阈值", "腐蚀", "膨胀", "开运算", "闭运算"])
            threshold = st.slider("全局阈值", 0, 255, 127)
            kernel = st.select_slider("结构元素大小", options=[3, 5, 7, 9], value=5)
            result = basic_enhance(image, method, threshold, kernel)
            explanation = "增强调整灰度分布；阈值生成二值图；形态学通过结构元素去除小噪声或连接区域。"
    st.caption(explanation)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 原图")
        st.image(bgr_to_rgb(image), use_container_width=True)
    with right:
        st.markdown("#### 处理结果")
        st.image(bgr_to_rgb(result), clamp=True, use_container_width=True)
        ok, encoded = cv2.imencode(".png", result)
        if ok:
            st.download_button("下载处理结果 PNG", encoded.tobytes(), "dip_result.png", "image/png")
    st.markdown("#### 原图灰度直方图")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.histogram(gray, bins=64, range=(0, 256))[0]
    st.line_chart({"灰度像素数": hist})
    m1, m2, m3 = st.columns(3)
    m1.metric("输入尺寸", f"{image.shape[1]} × {image.shape[0]}")
    m2.metric("结果尺寸", f"{result.shape[1]} × {result.shape[0]}")
    m3.metric("处理模块", module)


def retrieval_page(query: np.ndarray):
    st.subheader("齿轮 / 零件相似检索")
    st.write("将图库按 `dataset/类别名/图片` 组织。系统提取 Hu 矩、轮廓几何、HOG 和 HSV 颜色直方图，再按加权距离排序。")
    images = list_dataset_images()
    if not images:
        st.warning("尚无图库图片。请在项目目录创建 dataset/齿轮、dataset/螺母等文件夹并放入图片，然后刷新页面。")
        return
    if st.button("提取 / 更新图库特征"):
        with st.spinner("正在扫描图库并计算 Hu、HOG 与颜色特征…"):
            n = extract_gallery_features()
        st.success(f"已提取 {n} 张图库图片的特征，缓存保存在 cache/features.json。")
    cached_count = 0
    if FEATURE_CACHE.exists():
        try:
            cached_count = len(json.loads(FEATURE_CACHE.read_text(encoding="utf-8")).get("entries", []))
        except (OSError, json.JSONDecodeError):
            pass
    st.caption(f"当前图库：{len(images)} 张图片 · 特征缓存：{cached_count} 张")
    if not cached_count:
        st.info("请先点击“提取 / 更新图库特征”。添加或更换图库后也需重新提取。")
        return
    limit = st.slider("展示结果数", 1, min(10, len(images)), min(5, len(images)))
    with st.spinner("正在提取特征并比对图库…"):
        ranked = run_retrieval(query)[:limit]
    cols = st.columns(min(5, len(ranked)))
    for i, (dist, label, path, im) in enumerate(ranked):
        with cols[i % len(cols)]:
            st.image(bgr_to_rgb(im), use_container_width=True)
            st.markdown(f"**#{i+1} {label}**  \n距离：`{dist:.4f}`")


if __name__ == "__main__":
    main()
