# perf4：共享参数构造与批量贴图整体优化

本版把每帧的相机滤波、畸变网格、广告位透视网格、模糊核准备和最终 NumPy 输出一起纳入对照。旧对照同样每帧只构造一次共享参数，没有人为按 batch 重复计算。

## 本轮改动

- `paste_camera_filter.py`：保留邻域排序和现有运动估计回调，只计算需要的那一行 reflect 中值，不计算整个序列的滤波结果。
- `paste_shared_kernels.py`：一次 GPU kernel 构造畸变网格，10 轮逆畸变迭代在核内完成；没有 GPU 系数 `.tolist()`。每广告位一次 GPU kernel 生成透视网格，不创建齐次坐标大张量或执行小矩阵乘法。
- `paste_shared_params.py`：每次显式 prepare 读取当前 K、畸变系数；透视矩阵也是实时读取。多个 CPU 3×3 矩阵打包上传，透视网格使用当前素材的实际尺寸。仅对相同 sigma、回调、配置版本和设备的模糊核做有界缓存。
- `paste_gaussian_kernels.py`：可选横向模糊 + 纵向模糊/最终融合。先检查实际生成的二维核可分离，不能仅根据函数名假设核是 Gaussian。FP32 计算；小核不保证优于 perf3 密集融合路径。
- `paste_gpu_batch_adapter.py`：增加批量透视网格构造回调，识别带一维因子的 `MotionKernelSpec`；原调用签名及单组 HWC、多组 BHWC 的 NumPy uint8 输出不变。

`motion_rate_to_sigma`、`create_gaussian_kernel`、`compute_camera_motion_rate` 的具体实现尚未提供。本版生产接入直接调用你现有的方法，不替换 sigma 映射、Gaussian 截断半径或运动定义。基准中的这三个回调只是明确标注的合成示例。

## 两项正确性修正与性能开关分开

默认 `normalization="legacy"` 和 `camera_index_policy="legacy"` 保持所贴代码的行为，便于只比较性能。

1. 原网格总用 `2*x/(W-1)-1`，即使 `align_corners=False` 也不变。若 sampler 使用 False，会产生半像素量级的坐标偏差。显式选择 `normalization="matched"` 才使用与 sampler 对齐的公式：False 时 `(2*x+1)/W-1`，True 时原公式。两条路径的 warp/distortion sampler 配置必须分别与网格匹配。
2. 原 `current_index` 统计所有非当前帧的数量，当前帧存在时最终通常取排序后的最后一行。选择 `camera_index_policy="corrected"` 才按真正的 frame_id 取行。这是画面行为修正，不应混入仅比较速度的验收。

另外，原 `abs(k-frame_id)<5` 实际包含前后 4 帧，而不是注释中的 2 帧；本版保留半径 4。原长度为 3/4 的畸变系数仅使用 k1/k2，本版也保留；若希望按四系数模型使用切向项，应单独修改定义并验收。`inverse=True` 表示给未畸变的广告画布添加畸变；False 表示去畸变的反向采样网格。

透视矩阵明确采用 ROI 局部坐标，齐次除法保留 `z + 1e-8`。没有替换成旧 adapter 内置 helper 的有效 z 截断规则。legacy 归一化要求源图高宽均至少为 2；无效或发散的相机参数不会被静默修复。

## 接入生产

整包放在同一个 Python 模块目录。GPU worker 内初始化一次，保留已经跑通的 padding、alpha、布局、尺寸预处理和 sampler align_corners 配置：

```python
from utils_dir.shared_paste_example import setup_shared_paste, prepare_shared_paste_args
from utils_dir.paste_gpu_batch_adapter import paste_gpu_batch_adapter

# 初始化一次。先用 False，对照共享构造的收益；验证后再比较 True。
setup_shared_paste(
    self,
    self._paste_batch_adapter.options,
    use_separable=False,
    normalization="legacy",
    camera_index_policy="legacy",
    distortion_inverse=True,
    max_cached_sizes=3,
)

# 每帧一次：替换原先 compute_camera_filter / distortion_grid / kernel 的准备。
# 不要在此之前重复调用原滤波函数。
paste_args = prepare_shared_paste_args(
    self, frame_tensor, data_dict, frame_id, mask_canvas,
)

# 原调用签名不变。透视网格在 adapter 已确认实际素材尺寸后构造。
frame = paste_gpu_batch_adapter(
    self, frame_tensor, data_dict, frame_id, way_id, paste_args,
)
self.send_ad_img(raw_frame_id, frame, way_id, data_dict)
```

`setup_shared_paste` 需要已有 adapter 的 options，避免凭空猜测实际 sampler 配置。新 builder 默认没有 frame_id 结果缓存：同一帧 ID 的相机估计可能原地更新，不能只凭 ID 复用旧网格。

同设备、同分辨率、同 K/畸变/运动参数的多个 way 可以复用同一份 `paste_args`；各 way 的透视矩阵和源素材尺寸仍独立读取。mask 不同可使用 `way_args = (*paste_args[:4], mask_for_way, paste_args[5])`。

多设备时，CPU 相机滤波只做一次，再按设备构造 GPU 网格：

```python
from utils_dir.paste_camera_filter import compute_camera_filter_shared
camera_result = compute_camera_filter_shared(
    self, data_dict, frame_id, index_policy="legacy")
# 各设备分别调用它自己的 builder：
paste_args = builder.prepare(
    self, data_dict, frame_id, height=H, width=W, device=device,
    mask_canvas=mask_canvas, camera_result=camera_result)
```

参与 NumPy 中值滤波的历史系数需在 CPU；不隐式将 GPU 历史系数读回。w2c 保持原对象/设备传入现有 motion 回调，由回调负责处理；邻居不足时返回的原始系数也保持原设备。CPU 相机滤波仍由你的业务调用时机决定，不新增等待未来帧。

## Gaussian 缓存与可分离边界

核缓存按精确 sigma，不做量化。连续变化的 sigma 可能每帧未命中，基准默认覆盖这种情形。修改 Gaussian 配置但函数对象不变时，增加 `self.gaussian_kernel_cache_version`，或执行 `self._shared_paste_builder.clear_cache()`；不希望依赖回调纯函数约定时使用 `cache_kernels=False`。

`use_separable=True` 检查转换为 float32 后的实际二维核：非负、有限、外积重建满足 rtol=1e-6 / atol=1e-8。无法通过则明确报错；使用密集路径，不进行低秩近似。可分离路径只支持所有 batch/RGBA 共享因子、奇数长度 <=63；dense Triton 仍支持奇数高宽 <=15，较大的一般二维核用 PyTorch 模糊。

可分离卷积改变了浮点求和次序，不能保证所有输入逐字节相同。解释器测试最大差 1；目标 GPU 的完整输出也必须验证。该路径仍需要一张 RGBA 浮点中间图，因此以完整调用实测决定是否启用。

## 整体测试

```bat
python benchmark_shared_paste.py --version
python benchmark_shared_paste.py --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 --diagnose
python benchmark_shared_paste.py --device cuda:0 --batch 8 --slots 16 --ad-sizes 200x200 500x500 300x600 --diagnose
```

版本应为 `2026-10-20-perf4`。默认同轮比较以下三条完整路径，均包含独立的固定页 NumPy 输出：

| 字段 | 共享准备 | 贴图与模糊 |
|---|---|---|
| original_shared_plus_dense | 原 PyTorch 网格、SciPy 全序列滤波、每帧 Gaussian 生成 | perf3 密集融合核 |
| optimized_shared_plus_dense | 新共享准备 | 同一密集融合核 |
| optimized_shared_plus_separable | 新共享准备 | 经验证的可分离路径 |

关注 `shared_preparation_optimization_saving_ms`、`additional_separable_saving_ms` 和 `total_saving_vs_original_shared_dense_ms`；后者的基线不是之前纯贴图的 13.849ms。`--include-eager` 可增加原共享准备 + 当前 adapter eager 算子的对照。

将 `--diagnose` 换成 `--check-only` 只检查动态参数更新和最终输出。`--diagnose` 另测仅共享准备（包含所有透视网格），它与其他独立微基准不能直接相加。`--sigma-pattern repeated` 测试少量精确 sigma 复用，默认 `continuous` 测试持续变化；两种情况都应检查输出的核缓存命中统计。

测试 `--kernel-size 17` 等较大核时同时传 `--dense-blur-backend torch`。几何修正另跑 `--normalization matched`，滤波索引修正另跑 `--camera-index-policy corrected`；两者都让新旧对照采用相同语义。`--forward` 测试去畸变分支；`--align-corners` 为两种 sampler 同时启用 True，默认 False。

旧的 `benchmark_paste_gpu_batch.py` 仍可测试预先构造网格后的核心性能和尺寸缓存，但不包含这次新增的共享构造成本。

## 验证范围

本地没有 CUDA GPU，不能给出新增优化已节省多少毫秒的结论。已执行：

- 原 dense eager/auto CPU 回归，每套 24 个核心、48 个 adapter、8 个矩阵更新及特殊输出用例。
- 4,200 组相机单行中值与独立 SciPy 全序列滤波对照。
- 144 组共享畸变 Torch fallback 与独立 NumPy 公式对照，以及精确 sigma 缓存、回调和实时矩阵更新检查。
- 实际 Triton CPU interpreter：43 个畸变、27 个 ROI 网格用例最大网格差 0；44 个可分离卷积用例最大 uint8 差 1。
- 新端到端 benchmark 的三组 CPU 小尺寸配置，含三种非正方形素材、B1/B2、正反畸变和动态参数，最大网格差及 uint8 差均为 0。
- 最终集成检查覆盖空值 grid 回调、z=0 时的 z+eps、8 种布局解释、GPU 标记的 fallback/pose 原样传递，以及 batch/bucketed × B1/B2 可分离路由（uint8 差 0）。
- 新网格及可分离核共 12 种 SM89 离线编译配置。

解释器与离线编译不能替代 Windows/生产 GPU 的执行与性能验收。Gaussian 因子重建容差、浮点卷积求和次序和实际相机参数都可能影响最终像素，目标机运行脚本会先做动态输出校验，再计时。

CPU 相机滤波、动态 sigma/核尺寸和 NumPy 输出仍在 GPU 图外。GPU 几何核的矩阵/系数值变化不要求每帧重编译，但不能据此宣称整段流程已经可直接 CUDA Graph replay。

坐标与边界约定参考：
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
