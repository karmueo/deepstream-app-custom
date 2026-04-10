#!/usr/bin/env python3
"""使用 TensorRT 对视频执行 YOLO26 推理并导出带检测框的视频。"""

from __future__ import annotations

import argparse
import ctypes
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import tensorrt as trt
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TensorBindingInfo:
    """TensorRT 绑定信息。"""

    name: str  # 绑定名称
    mode: trt.TensorIOMode  # 绑定方向
    dtype: np.dtype  # 绑定数据类型
    shape: tuple[int, ...]  # 绑定形状


@dataclass
class FramePreprocessInfo:
    """单帧预处理结果与回映射信息。"""

    tensor: np.ndarray  # 网络输入张量
    ratio: float  # 缩放比例
    pad_left: int  # 左侧填充像素
    pad_top: int  # 顶部填充像素
    original_width: int  # 原始帧宽度
    original_height: int  # 原始帧高度


@dataclass
class Detection:
    """单个检测框结果。"""

    left: int  # 左上角 x 坐标
    top: int  # 左上角 y 坐标
    right: int  # 右下角 x 坐标
    bottom: int  # 右下角 y 坐标
    score: float  # 置信度
    class_id: int  # 类别 ID


class CudaRuntime:
    """基于 libcudart 的轻量 CUDA 运行时封装。"""

    CUDA_MEMCPY_HOST_TO_DEVICE = 1
    CUDA_MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        """初始化 CUDA 运行时接口。

        Args:
            None.

        Returns:
            None.
        """
        lib_name = self._resolve_cudart_library()  # CUDA runtime 库名
        self._lib = ctypes.CDLL(lib_name)  # CUDA runtime 动态库
        self._configure_prototypes()

    def _resolve_cudart_library(self) -> str:
        """解析 libcudart 的实际路径。

        Args:
            None.

        Returns:
            str: CUDA runtime 动态库路径。
        """
        candidates = [find_library("cudart")]  # 候选 CUDA runtime 库路径
        candidates.extend(
            [
                "libcudart.so.12",
                "libcudart.so",
                "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so.12",
                "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
                "/usr/local/cuda/lib64/libcudart.so.12",
                "/usr/lib/aarch64-linux-gnu/libcudart.so.12",
            ]
        )

        for candidate in candidates:
            if not candidate:
                continue
            try:
                ctypes.CDLL(candidate)
                return candidate
            except OSError:
                continue

        raise RuntimeError("无法加载 libcudart.so，请确认 CUDA 运行时已安装且可被系统找到。")

    def _configure_prototypes(self) -> None:
        """配置 CUDA runtime 函数原型。

        Args:
            None.

        Returns:
            None.
        """
        self._lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._lib.cudaMalloc.restype = ctypes.c_int

        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int

        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int

        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cudaStreamCreate.restype = ctypes.c_int

        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int

        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int

    @staticmethod
    def _check(status: int, action: str) -> None:
        """检查 CUDA API 返回值。

        Args:
            status (int): CUDA API 返回码。
            action (str): 当前执行的动作描述。

        Returns:
            None.
        """
        if status != 0:
            raise RuntimeError(f"CUDA 调用失败: {action}, 错误码={status}")

    def malloc(self, size_bytes: int) -> ctypes.c_void_p:
        """申请 GPU 显存。

        Args:
            size_bytes (int): 申请的字节数。

        Returns:
            ctypes.c_void_p: GPU 显存指针。
        """
        device_ptr = ctypes.c_void_p()  # GPU 内存指针
        self._check(
            self._lib.cudaMalloc(ctypes.byref(device_ptr), ctypes.c_size_t(size_bytes)),
            f"cudaMalloc({size_bytes})",
        )
        return device_ptr

    def free(self, device_ptr: ctypes.c_void_p) -> None:
        """释放 GPU 显存。

        Args:
            device_ptr (ctypes.c_void_p): GPU 显存指针。

        Returns:
            None.
        """
        if device_ptr:
            self._check(self._lib.cudaFree(device_ptr), "cudaFree")

    def create_stream(self) -> ctypes.c_void_p:
        """创建 CUDA stream。

        Args:
            None.

        Returns:
            ctypes.c_void_p: CUDA stream 句柄。
        """
        stream = ctypes.c_void_p()  # CUDA stream 句柄
        self._check(self._lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        """销毁 CUDA stream。

        Args:
            stream (ctypes.c_void_p): CUDA stream 句柄。

        Returns:
            None.
        """
        if stream:
            self._check(self._lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def memcpy_htod_async(self, device_ptr: ctypes.c_void_p, host_array: np.ndarray, stream: ctypes.c_void_p) -> None:
        """将主机数据异步拷贝到设备。

        Args:
            device_ptr (ctypes.c_void_p): 目标设备地址。
            host_array (np.ndarray): 源主机数组。
            stream (ctypes.c_void_p): CUDA stream。

        Returns:
            None.
        """
        host_ptr = ctypes.c_void_p(host_array.ctypes.data)  # 主机数据指针
        self._check(
            self._lib.cudaMemcpyAsync(
                device_ptr,
                host_ptr,
                ctypes.c_size_t(host_array.nbytes),
                self.CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync H2D",
        )

    def memcpy_dtoh_async(self, host_array: np.ndarray, device_ptr: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
        """将设备数据异步拷贝到主机。

        Args:
            host_array (np.ndarray): 目标主机数组。
            device_ptr (ctypes.c_void_p): 源设备地址。
            stream (ctypes.c_void_p): CUDA stream。

        Returns:
            None.
        """
        host_ptr = ctypes.c_void_p(host_array.ctypes.data)  # 主机数据指针
        self._check(
            self._lib.cudaMemcpyAsync(
                host_ptr,
                device_ptr,
                ctypes.c_size_t(host_array.nbytes),
                self.CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync D2H",
        )

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        """同步 CUDA stream。

        Args:
            stream (ctypes.c_void_p): CUDA stream。

        Returns:
            None.
        """
        self._check(self._lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


def resolve_video_source(source: str) -> str:
    """将视频输入地址规范化为 OpenCV 可识别的形式。

    Args:
        source (str): 用户输入的视频地址。

    Returns:
        str: 规范化后的输入地址。
    """
    parsed = urlparse(source)  # 解析输入地址
    if parsed.scheme == "file":
        local_path = Path(unquote(parsed.path)).expanduser()  # 本地文件路径
        return str(local_path.resolve())
    if parsed.scheme in {"rtsp", "http", "https"}:
        return source

    local_path = Path(source).expanduser()  # 本地视频路径
    if local_path.exists():
        return str(local_path.resolve())
    return source


def load_labels(label_path: Path) -> list[str]:
    """加载类别标签文件。

    Args:
        label_path (Path): 标签文件路径。

    Returns:
        list[str]: 标签名称列表。
    """
    if not label_path.is_file():
        return []

    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]  # 标签列表
    return [label for label in labels if label]


def find_cjk_font() -> Path | None:
    """查找系统中可用于绘制中文的字体文件。

    Args:
        None.

    Returns:
        Path | None: 字体文件路径，找不到则返回 None。
    """
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    try:
        import subprocess

        result = subprocess.run(
            ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"],
            check=True,
            capture_output=True,
            text=True,
        )
        font_path = Path(result.stdout.strip())  # 字体路径
        if font_path.is_file():
            return font_path
    except Exception:
        pass

    return None


def clamp_int(value: float, min_value: int, max_value: int) -> int:
    """将数值裁剪到整数区间。

    Args:
        value (float): 待裁剪数值。
        min_value (int): 最小值。
        max_value (int): 最大值。

    Returns:
        int: 裁剪后的整数。
    """
    return int(max(min_value, min(max_value, round(value))))


class TensorRTVideoInferencer:
    """纯 TensorRT 视频推理器。"""

    def __init__(
        self,
        engine_path: Path,
        label_path: Path,
        conf_threshold: float,
        user_batch: int | None = None,
    ) -> None:
        """初始化 TensorRT 推理器。

        Args:
            engine_path (Path): TensorRT engine 文件路径。
            label_path (Path): 类别标签文件路径。
            conf_threshold (float): 置信度阈值。
            user_batch (int | None): 用户指定 batch 大小。

        Returns:
            None.
        """
        self.logger = trt.Logger(trt.Logger.WARNING)  # TensorRT 日志器
        self.runtime = trt.Runtime(self.logger)  # TensorRT 运行时
        self.cuda = CudaRuntime()  # CUDA 运行时
        self.engine_path = engine_path  # TensorRT 引擎路径
        self.conf_threshold = conf_threshold  # 检测置信度阈值
        self.labels = load_labels(label_path)  # 类别标签列表
        self.font_path = find_cjk_font()  # 中文字体路径
        self.font_size = 20  # 标签字体大小
        self.font = self._load_font()  # PIL 字体对象
        self.engine = self._load_engine(engine_path)  # TensorRT 引擎对象
        self.context = self.engine.create_execution_context()  # 执行上下文
        self.input_binding, self.output_binding = self._inspect_bindings()  # 输入/输出绑定
        self.input_shape = self._resolve_input_shape(user_batch)  # 推理输入形状
        self.output_shape = self._resolve_output_shape()  # 推理输出形状
        self.stream = self.cuda.create_stream()  # CUDA stream
        self.input_host = np.empty(self.input_shape, dtype=self.input_binding.dtype)  # 主机输入缓冲区
        self.output_host = np.empty(self.output_shape, dtype=self.output_binding.dtype)  # 主机输出缓冲区
        self.input_device = self.cuda.malloc(self.input_host.nbytes)  # 设备输入缓冲区
        self.output_device = self.cuda.malloc(self.output_host.nbytes)  # 设备输出缓冲区
        self.context.set_tensor_address(self.input_binding.name, int(self.input_device.value))  # 输入地址
        self.context.set_tensor_address(self.output_binding.name, int(self.output_device.value))  # 输出地址

    def close(self) -> None:
        """释放推理器资源。

        Args:
            None.

        Returns:
            None.
        """
        self.cuda.free(self.input_device)
        self.cuda.free(self.output_device)
        self.cuda.destroy_stream(self.stream)

    def _load_engine(self, engine_path: Path) -> trt.ICudaEngine:
        """加载 TensorRT engine。

        Args:
            engine_path (Path): TensorRT engine 文件路径。

        Returns:
            trt.ICudaEngine: 反序列化后的引擎对象。
        """
        if not engine_path.is_file():
            raise FileNotFoundError(f"找不到 engine 文件: {engine_path}")

        engine_bytes = engine_path.read_bytes()  # engine 序列化数据
        engine = self.runtime.deserialize_cuda_engine(engine_bytes)  # 反序列化引擎
        if engine is None:
            raise RuntimeError(f"无法加载 TensorRT engine: {engine_path}")
        return engine

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """加载用于中文显示的字体。

        Args:
            None.

        Returns:
            ImageFont.FreeTypeFont | ImageFont.ImageFont: 字体对象。
        """
        if self.font_path is None:
            return ImageFont.load_default()

        try:
            return ImageFont.truetype(str(self.font_path), self.font_size)
        except Exception:
            return ImageFont.load_default()

    def _inspect_bindings(self) -> tuple[TensorBindingInfo, TensorBindingInfo]:
        """解析 engine 的输入输出绑定。

        Args:
            None.

        Returns:
            tuple[TensorBindingInfo, TensorBindingInfo]: 输入与输出绑定信息。
        """
        bindings: list[TensorBindingInfo] = []  # 绑定信息列表
        for index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(index)  # Tensor 名称
            tensor_mode = self.engine.get_tensor_mode(tensor_name)  # Tensor 方向
            tensor_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(tensor_name)))  # Tensor dtype
            tensor_shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(tensor_name))  # Tensor shape
            bindings.append(
                TensorBindingInfo(
                    name=tensor_name,
                    mode=tensor_mode,
                    dtype=tensor_dtype,
                    shape=tensor_shape,
                )
            )

        input_bindings = [binding for binding in bindings if binding.mode == trt.TensorIOMode.INPUT]  # 输入绑定
        output_bindings = [binding for binding in bindings if binding.mode == trt.TensorIOMode.OUTPUT]  # 输出绑定

        if len(input_bindings) != 1:
            raise RuntimeError(f"当前脚本仅支持 1 个输入 Tensor，但检测到 {len(input_bindings)} 个。")
        if len(output_bindings) != 1:
            raise RuntimeError(f"当前脚本仅支持 1 个输出 Tensor，但检测到 {len(output_bindings)} 个。")

        return input_bindings[0], output_bindings[0]

    def _resolve_input_shape(self, user_batch: int | None) -> tuple[int, ...]:
        """解析实际推理输入形状。

        Args:
            user_batch (int | None): 用户指定的 batch 大小。

        Returns:
            tuple[int, ...]: 可用于推理的输入形状。
        """
        engine_shape = list(self.input_binding.shape)  # 引擎定义的输入形状
        if not engine_shape:
            raise RuntimeError("engine 输入形状为空。")

        if engine_shape[0] > 0:
            batch_size = engine_shape[0]  # 静态 batch 大小
            if user_batch is not None and user_batch != batch_size:
                print(
                    f"[WARN] engine 是静态 batch={batch_size}，将忽略用户指定的 batch={user_batch}。"
                )
            input_shape = tuple(engine_shape)  # 最终输入形状
        else:
            batch_size = user_batch or 1  # 动态 batch 大小
            input_shape = tuple([batch_size] + engine_shape[1:])  # 动态 batch 输入形状

        self.context.set_input_shape(self.input_binding.name, input_shape)
        resolved_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.input_binding.name))
        if any(dim < 0 for dim in resolved_shape):
            raise RuntimeError(f"输入形状解析失败: {resolved_shape}")
        return resolved_shape

    def _resolve_output_shape(self) -> tuple[int, ...]:
        """解析实际推理输出形状。

        Args:
            None.

        Returns:
            tuple[int, ...]: 输出张量形状。
        """
        resolved_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_binding.name))
        if not resolved_shape or any(dim < 0 for dim in resolved_shape):
            raise RuntimeError(f"输出形状解析失败: {resolved_shape}")
        return resolved_shape

    @property
    def batch_size(self) -> int:
        """获取当前推理 batch 大小。

        Args:
            None.

        Returns:
            int: batch 大小。
        """
        return int(self.input_shape[0])

    @property
    def input_height(self) -> int:
        """获取网络输入高度。

        Args:
            None.

        Returns:
            int: 输入高度。
        """
        return int(self.input_shape[2])

    @property
    def input_width(self) -> int:
        """获取网络输入宽度。

        Args:
            None.

        Returns:
            int: 输入宽度。
        """
        return int(self.input_shape[3])

    def preprocess_frame(self, frame: np.ndarray) -> FramePreprocessInfo:
        """对单帧视频做 letterbox 预处理。

        Args:
            frame (np.ndarray): 输入 BGR 帧。

        Returns:
            FramePreprocessInfo: 预处理后的网络输入与回映射信息。
        """
        original_height, original_width = frame.shape[:2]  # 原始帧宽高
        scale = min(self.input_width / original_width, self.input_height / original_height)  # 缩放比例
        resized_width = int(round(original_width * scale))  # 缩放后宽度
        resized_height = int(round(original_height * scale))  # 缩放后高度

        resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)  # 缩放帧

        pad_width = self.input_width - resized_width  # 水平填充总像素
        pad_height = self.input_height - resized_height  # 垂直填充总像素
        pad_left = pad_width // 2  # 左侧填充像素
        pad_right = pad_width - pad_left  # 右侧填充像素
        pad_top = pad_height // 2  # 顶部填充像素
        pad_bottom = pad_height - pad_top  # 底部填充像素

        padded_frame = cv2.copyMakeBorder(
            resized_frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )  # 中心填充后的图像
        rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
        normalized_frame = rgb_frame.astype(np.float32) / 255.0  # 归一化到 [0,1]
        chw_frame = np.transpose(normalized_frame, (2, 0, 1))  # HWC 转 CHW
        tensor = np.ascontiguousarray(chw_frame.astype(self.input_binding.dtype, copy=False))  # 网络输入张量

        return FramePreprocessInfo(
            tensor=tensor,
            ratio=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            original_width=original_width,
            original_height=original_height,
        )

    def infer_batch(self, batch_tensor: np.ndarray) -> np.ndarray:
        """执行一批次推理。

        Args:
            batch_tensor (np.ndarray): 形状为 [B, C, H, W] 的输入张量。

        Returns:
            np.ndarray: 模型输出张量。
        """
        if batch_tensor.shape != self.input_shape:
            raise ValueError(f"输入 batch 形状不匹配，期望 {self.input_shape}，实际 {batch_tensor.shape}")

        batch_tensor = np.ascontiguousarray(batch_tensor.astype(self.input_binding.dtype, copy=False))  # 输入批数据
        self.cuda.memcpy_htod_async(self.input_device, batch_tensor, self.stream)
        self.context.execute_async_v3(stream_handle=int(self.stream.value))
        self.cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.cuda.synchronize(self.stream)
        return np.array(self.output_host, copy=True)

    def decode_sample_output(
        self,
        sample_output: np.ndarray,
        frame_info: FramePreprocessInfo,
    ) -> list[Detection]:
        """将单帧输出转换为原始视频坐标系下的检测框。

        Args:
            sample_output (np.ndarray): 单帧模型输出。
            frame_info (FramePreprocessInfo): 单帧预处理回映射信息。

        Returns:
            list[Detection]: 检测结果列表。
        """
        output_array = np.asarray(sample_output)  # 单帧输出数组

        if output_array.ndim == 1:
            if output_array.size % 6 != 0:
                raise RuntimeError(f"输出长度 {output_array.size} 不能被 6 整除，无法解析检测框。")
            output_array = output_array.reshape(-1, 6)
        elif output_array.ndim == 2:
            if output_array.shape[-1] == 6:
                pass
            elif output_array.shape[0] == 6:
                output_array = output_array.transpose(1, 0)
            else:
                raise RuntimeError(f"无法识别的单帧输出形状: {output_array.shape}")
        else:
            raise RuntimeError(f"无法识别的单帧输出维度: {output_array.shape}")

        output_array = output_array.astype(np.float32, copy=False)  # 转成 float32 便于后处理
        detections: list[Detection] = []  # 单帧检测结果

        for row in output_array:
            if row.size < 6:
                continue

            left, top, right, bottom, score, class_id = row[:6]  # 单条检测结果
            if score < self.conf_threshold:
                continue

            mapped_left = (left - frame_info.pad_left) / frame_info.ratio  # 映射回原图坐标
            mapped_top = (top - frame_info.pad_top) / frame_info.ratio  # 映射回原图坐标
            mapped_right = (right - frame_info.pad_left) / frame_info.ratio  # 映射回原图坐标
            mapped_bottom = (bottom - frame_info.pad_top) / frame_info.ratio  # 映射回原图坐标

            clipped_left = clamp_int(mapped_left, 0, frame_info.original_width - 1)  # 裁剪后的左边界
            clipped_top = clamp_int(mapped_top, 0, frame_info.original_height - 1)  # 裁剪后的上边界
            clipped_right = clamp_int(mapped_right, 0, frame_info.original_width - 1)  # 裁剪后的右边界
            clipped_bottom = clamp_int(mapped_bottom, 0, frame_info.original_height - 1)  # 裁剪后的下边界

            if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
                continue

            detections.append(
                Detection(
                    left=clipped_left,
                    top=clipped_top,
                    right=clipped_right,
                    bottom=clipped_bottom,
                    score=float(score),
                    class_id=int(round(class_id)),
                )
            )

        return detections

    def _label_text(self, detection: Detection) -> str:
        """生成单个检测框的显示文本。

        Args:
            detection (Detection): 检测结果。

        Returns:
            str: 可显示的标签文本。
        """
        if 0 <= detection.class_id < len(self.labels):
            class_name = self.labels[detection.class_id]  # 类别名称
        else:
            class_name = f"class_{detection.class_id}"  # 兜底类别名称
        return f"{class_name} {detection.score:.2f}"

    def draw_detections(self, frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
        """在视频帧上绘制检测结果。

        Args:
            frame (np.ndarray): 原始 BGR 帧。
            detections (Iterable[Detection]): 检测结果集合。

        Returns:
            np.ndarray: 绘制后的 BGR 帧。
        """
        canvas = frame.copy()  # 待绘制图像
        detections_list = list(detections)  # 检测结果列表

        for detection in detections_list:
            color = self._color_for_class(detection.class_id)  # 框颜色
            cv2.rectangle(
                canvas,
                (detection.left, detection.top),
                (detection.right, detection.bottom),
                color,
                2,
                lineType=cv2.LINE_AA,
            )

        if not detections_list:
            return canvas

        if self.font_path is not None:
            canvas = self._draw_text_with_pil(canvas, detections_list)
        else:
            canvas = self._draw_text_with_cv2(canvas, detections_list)

        return canvas

    def _color_for_class(self, class_id: int) -> tuple[int, int, int]:
        """生成类别对应的框颜色。

        Args:
            class_id (int): 类别 ID。

        Returns:
            tuple[int, int, int]: BGR 颜色值。
        """
        base = abs(class_id) + 1  # 颜色基数
        blue = (37 * base) % 255  # 蓝色分量
        green = (17 * base) % 255  # 绿色分量
        red = (97 * base) % 255  # 红色分量
        return blue, green, red

    def _draw_text_with_cv2(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """使用 OpenCV 绘制文本。

        Args:
            frame (np.ndarray): 原始 BGR 帧。
            detections (list[Detection]): 检测结果集合。

        Returns:
            np.ndarray: 绘制后的 BGR 帧。
        """
        canvas = frame.copy()  # OpenCV 文本绘制帧
        for detection in detections:
            label_text = self._label_text(detection)  # 文本标签
            origin_x = detection.left  # 文本起始 x 坐标
            origin_y = max(0, detection.top - 6)  # 文本起始 y 坐标
            cv2.putText(
                canvas,
                label_text,
                (origin_x, origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        return canvas

    def _draw_text_with_pil(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """使用 PIL 绘制中文文本。

        Args:
            frame (np.ndarray): 原始 BGR 帧。
            detections (list[Detection]): 检测结果集合。

        Returns:
            np.ndarray: 绘制后的 BGR 帧。
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
        pil_image = Image.fromarray(rgb_image)  # PIL 图像对象
        draw = ImageDraw.Draw(pil_image)  # PIL 绘图器

        for detection in detections:
            label_text = self._label_text(detection)  # 文本标签
            color = self._color_for_class(detection.class_id)  # 框颜色
            rgb_color = (color[2], color[1], color[0])  # RGB 颜色值
            text_bbox = draw.textbbox((0, 0), label_text, font=self.font)  # 文本边界框
            text_width = text_bbox[2] - text_bbox[0]  # 文本宽度
            text_height = text_bbox[3] - text_bbox[1]  # 文本高度

            text_x = detection.left  # 文本左上角 x 坐标
            text_y = max(0, detection.top - text_height - 6)  # 文本左上角 y 坐标
            background_right = min(pil_image.width - 1, text_x + text_width + 8)  # 背景右边界
            background_bottom = min(pil_image.height - 1, text_y + text_height + 6)  # 背景下边界

            draw.rectangle(
                [(text_x, text_y), (background_right, background_bottom)],
                fill=rgb_color,
            )
            draw.text((text_x + 4, text_y + 2), label_text, fill=(255, 255, 255), font=self.font)

        result_rgb = np.array(pil_image)  # PIL 转回 numpy
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    def process_video(self, source: str, output_path: Path) -> Path:
        """处理视频并导出检测结果视频。

        Args:
            source (str): 视频输入地址。
            output_path (Path): 输出视频路径。

        Returns:
            Path: 输出视频文件路径。
        """
        normalized_source = resolve_video_source(source)  # 规范化的视频输入地址
        capture = cv2.VideoCapture(normalized_source)  # 视频捕获器
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频输入: {source}")

        input_fps = capture.get(cv2.CAP_PROP_FPS)  # 输入视频帧率
        fps = input_fps if input_fps and input_fps > 0 else 25.0  # 输出帧率
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 输入总帧数，部分流可能为 0
        total_frames_text = str(total_frames) if total_frames > 0 else "未知"  # 总帧数显示文本

        print(
            "[INFO] 启动视频推理",
            f"\n  engine: {self.engine_path}",
            f"\n  source: {normalized_source}",
            f"\n  output: {output_path}",
            f"\n  batch : {self.batch_size}",
            f"\n  input : {self.input_shape}",
            f"\n  output: {self.output_shape}",
            f"\n  fps   : {fps:.2f}",
            f"\n  frames: {total_frames_text}",
            flush=True,
        )

        ret, first_frame = capture.read()  # 先读取第一帧，确保可以获取分辨率
        if not ret:
            raise RuntimeError(f"无法从输入源读取第一帧: {source}")

        output_height, output_width = first_frame.shape[:2]  # 输出视频宽高

        output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建输出目录
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码器
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))  # 视频写入器
        if not writer.isOpened():
            raise RuntimeError(f"无法创建输出视频: {output_path}")

        batch_frames: list[np.ndarray] = [first_frame]  # 当前批次原始帧
        batch_preprocessed: list[FramePreprocessInfo] = [self.preprocess_frame(first_frame)]  # 当前批次预处理数据
        frame_index = 1  # 已处理帧编号
        end_of_stream = False  # 是否已到达视频末尾
        batch_count = 0  # 已处理 batch 数
        start_time = time.perf_counter()  # 推理开始时间
        last_log_time = start_time  # 最近一次日志时间
        last_log_frames = 0  # 最近一次日志对应的已写入帧数

        try:
            while not end_of_stream or batch_frames:
                batch_start = time.perf_counter()  # 当前 batch 起始时间
                while not end_of_stream and len(batch_frames) < self.batch_size:
                    ret, frame = capture.read()  # 继续读取视频帧
                    if not ret:
                        end_of_stream = True
                        break
                    batch_frames.append(frame)  # 保存原始帧
                    batch_preprocessed.append(self.preprocess_frame(frame))  # 保存预处理信息
                    frame_index += 1

                real_count = len(batch_frames)  # 当前批次真实帧数
                while len(batch_frames) < self.batch_size:
                    batch_frames.append(batch_frames[-1].copy())  # 复制最后一帧补齐 batch
                    batch_preprocessed.append(batch_preprocessed[-1])  # 复制最后一帧的预处理信息

                batch_tensor = np.stack([item.tensor for item in batch_preprocessed], axis=0)  # 批输入张量
                output_tensor = self.infer_batch(batch_tensor)  # 模型输出

                for batch_index in range(real_count):
                    per_frame_output = output_tensor[batch_index]  # 单帧输出
                    detections = self.decode_sample_output(per_frame_output, batch_preprocessed[batch_index])  # 解码检测结果
                    rendered_frame = self.draw_detections(batch_frames[batch_index], detections)  # 绘制检测框
                    writer.write(rendered_frame)  # 写入输出视频
                    last_log_frames += 1  # 累计已写入帧数

                batch_frames.clear()
                batch_preprocessed.clear()
                batch_count += 1  # 累计已处理 batch 数

                now = time.perf_counter()  # 当前时间点
                batch_elapsed = now - batch_start  # 当前 batch 耗时
                total_elapsed = now - start_time  # 总耗时
                interval_elapsed = now - last_log_time  # 最近一次日志到现在的间隔
                instant_fps = last_log_frames / interval_elapsed if interval_elapsed > 0 else 0.0  # 最近区间 FPS
                avg_fps = frame_index / total_elapsed if total_elapsed > 0 else 0.0  # 平均 FPS
                progress_text = f"{frame_index}/{total_frames_text}" if total_frames > 0 else f"{frame_index}"  # 进度文本
                print(
                    f"[PROGRESS] batch={batch_count} "
                    f"frames={progress_text} "
                    f"batch_time={batch_elapsed:.3f}s "
                    f"inst_fps={instant_fps:.2f} "
                    f"avg_fps={avg_fps:.2f}",
                    flush=True,
                )
                last_log_time = now  # 更新日志时间点
                last_log_frames = 0  # 重置区间帧计数

            print(f"[INFO] 视频推理完成，共处理 {frame_index} 帧，输出文件: {output_path}")
            return output_path
        finally:
            capture.release()
            writer.release()


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    Args:
        None.

    Returns:
        argparse.ArgumentParser: 命令行参数解析器。
    """
    parser = argparse.ArgumentParser(description="纯 TensorRT 视频推理脚本")
    parser.add_argument("--engine", required=True, help="TensorRT engine 文件路径")
    parser.add_argument("--source", required=True, help="输入视频地址，支持本地路径或 file:// / rtsp:// 地址")
    parser.add_argument(
        "--output",
        default=None,
        help="输出视频路径，默认根据输入文件名生成 detected_output.mp4",
    )
    parser.add_argument(
        "--labels",
        default=str((Path(__file__).resolve().parents[1] / "configs" / "labels2.txt")),
        help="类别标签文件路径",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="检测置信度阈值",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="动态 batch engine 的 batch 大小；静态 batch engine 会忽略该参数",
    )
    return parser


def resolve_output_path(source: str, output_arg: str | None) -> Path:
    """解析输出视频路径。

    Args:
        source (str): 输入视频地址。
        output_arg (str | None): 用户指定的输出路径。

    Returns:
        Path: 输出视频路径。
    """
    if output_arg:
        return Path(output_arg).expanduser().resolve()

    parsed = urlparse(source)  # 输入地址解析结果
    if parsed.scheme == "file":
        source_path = Path(unquote(parsed.path)).expanduser()  # 本地输入路径
    elif parsed.scheme in {"rtsp", "http", "https"}:
        return Path.cwd() / "detected_output.mp4"
    else:
        source_path = Path(source).expanduser()  # 输入源路径

    if source_path.suffix:
        output_name = f"{source_path.stem}_detected.mp4"  # 默认输出文件名
    else:
        output_name = "detected_output.mp4"  # 默认输出文件名
    return Path.cwd() / output_name


def main() -> None:
    """程序入口，完成视频推理和结果导出。

    Args:
        None.

    Returns:
        None.
    """
    parser = build_arg_parser()  # 命令行参数解析器
    args = parser.parse_args()  # 命令行参数

    engine_path = Path(args.engine).expanduser().resolve()  # TensorRT 引擎路径
    label_path = Path(args.labels).expanduser().resolve()  # 标签文件路径
    source_path = args.source  # 视频输入地址
    output_path = resolve_output_path(source_path, args.output)  # 输出视频路径
    conf_thresh = float(args.conf_thresh)  # 置信度阈值
    user_batch = args.batch  # 用户指定 batch 大小

    print(
        "[INFO] 准备初始化 TensorRT 推理器",
        f"\n  engine: {engine_path}",
        f"\n  source: {source_path}",
        f"\n  output: {output_path}",
        f"\n  labels: {label_path}",
        flush=True,
    )

    inferencer = TensorRTVideoInferencer(
        engine_path=engine_path,
        label_path=label_path,
        conf_threshold=conf_thresh,
        user_batch=user_batch,
    )  # TensorRT 推理器

    try:
        print("[INFO] 推理器初始化完成，开始视频推理", flush=True)
        inferencer.process_video(source_path, output_path)
    finally:
        inferencer.close()
        print("[INFO] 资源已释放", flush=True)


if __name__ == "__main__":
    main()
