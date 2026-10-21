paste_gpu_batch 完整优化包 — 2026-10-21-perf4.1

perf4.1 修复 Python 3.9 导入时 tuple[...] | None 类型注解求值错误。
paste_gpu_batch.py 补充延迟注解，两个 benchmark 打印 Python 版本和解释器路径。
GPU 算法、缓存及输出不变；整包版本统一更新，不需要为本问题升级运行环境。

本版保留 perf3 密集模糊融合与多尺寸缓存，新增共享参数构造优化。
接入步骤和兼容性说明见 SHARED_PIPELINE_README.md，示例为 shared_paste_example.py。
所有 .py 文件放在同一模块目录；整包更新后重启 worker。不要只替换 benchmark。
每个 GPU worker 初始化一次、预热后使用，不在逐帧循环里重建 adapter/builder。

文件
  paste_gpu_batch.py             batch/bucketed 核心与可选 separable 模糊路由
  paste_gpu_batch_adapter.py     原接口、素材缓存、输出和批量网格构造回调
  paste_gpu_kernels.py           perf3 Triton 贴图、密集模糊/融合等内核
  paste_shared_types.py          MotionKernelSpec 显式核因子协议
  paste_shared_kernels.py        畸变与 ROI 网格融合内核
  paste_shared_params.py         共享 prepare、Gaussian 缓存、实际素材尺寸适配
  paste_camera_filter.py         单行 reflect 中值，调用现有 motion 方法
  paste_gaussian_kernels.py      可选可分离模糊与最终融合内核
  shared_paste_example.py        初始化与逐帧 prepare 示例
  benchmark_shared_paste.py      包含共享构造的动态完整调用对照
  benchmark_paste_gpu_batch.py   原核心/adapter、多尺寸缓存及诊断测试

依赖与版本
  沿用现有 PyTorch/CUDA 与匹配的 Triton、NumPy。
  新 benchmark 的独立旧版滤波参考另外依赖 SciPy。
  不要用开发检查环境的 PyTorch/Triton 版本覆盖生产环境。
  python benchmark_shared_paste.py --version
  python benchmark_paste_gpu_batch.py --version
  两个 benchmark 和 adapter 都应为 2026-10-21-perf4.1。
  --version / --help 无需GPU；正式CLI测速要求CUDA。

新增共享构造整体测试
  python benchmark_shared_paste.py --device cuda:0 --batch 8 --ad-sizes 200x200 500x500 --diagnose
  python benchmark_shared_paste.py --device cuda:0 --batch 8 --ad-sizes 200x200 500x500 300x600 --diagnose
  同次比较原共享准备+dense、新共享准备+同一dense、新共享准备+separable。
  都只每帧准备一次共享参数，都返回独立固定页 NumPy。
  K、畸变系数、单应矩阵原地更新；默认 sigma 连续变化。
  --sigma-pattern repeated 单独测精确sigma复用；输出实际缓存命中统计。
  --check-only 只校验；--include-eager 增加 eager 合成参考。
  大于15的奇数核加 --dense-blur-backend torch，例如 --kernel-size 17。
  新基线包括共享构造，不能和旧纯adapter的13.849ms基线直接相减。
  三个运动/sigma/Gaussian回调在benchmark中明确为合成示例，生产保留原方法。

保留的核心/adapter 多尺寸测试
  python benchmark_paste_gpu_batch.py --device cuda:0 --batch 8 --blur-backend triton --diagnose --ad-sizes 200x200 500x500
  python benchmark_paste_gpu_batch.py --device cuda:0 --batch 8 --blur-backend triton --diagnose --ad-sizes 200x200 500x500 300x600
  python benchmark_paste_gpu_batch.py --device cuda:0 --batch 8 --blur-backend triton --check-only --ad-sizes 200x200 500x500 300x600
  广告位按列表循环分配。HxW是源素材高x宽，不是bbox。
  同编号在batch各组尺寸相同，不同编号可不同。
  16广告位时两尺寸各8个，三尺寸6/5/5个；非方形也支持。
  默认缓存容量匹配实际尺寸数；--max-cached-sizes 2可测三尺寸的淘汰成本。
  fixture输出尺寸分配、bbox、容量和重复调用命中；足够容量时16次命中/0次未命中。

历史性能（用户实测，不是本次共享构造的结果）
  Windows / RTX4090D / PyTorch2.6+cu126 / B8 / 1080p / 16广告位 / 9x9核：
  perf3 eager adapter 13.849ms，dense Triton+固定页输出6.047ms，节省7.802ms。
  GPU利用率50%不能单独定位瓶颈；新优化需同负载实测。

输出与缓存
  默认 np.uint8：单组[H,W,3]，多组[B,H,W,3]；squeeze_single=False保留B1维。
  缓存按实际/显式resize后的(H,W)区分，不因ad_h_w不符而静默resize。
  三尺寸初始化设max_cached_sizes=3。修改已有options后需invalidate_assets()一次。
  不要每帧清缓存；素材原地修改须按既有接口失效/更新版本，避免旧素材。
  pinned_output=True每次结果独立，下一帧不会覆写下游仍持有的NumPy。
  无额外NumPy复制，等GPU结果可读后才返回；输出积压增加固定页内存占用。
  return_numpy=False仅用于GPU下游/诊断；send_ad_img保持True。

加速与语义
  acceleration=eager/auto/triton；triton不满足条件时明确报错。
  blur_backend=torch保留PyTorch模糊；triton密集融合支持奇数高宽<=15。
  separable需要验证过的共享(ky,kx)，奇数长度<=63；不静默低秩近似。
  默认仍eager、torch模糊和普通输出；示例才显式启用加速/固定页。
  不新增FP16/TF32；求和次序可能导致uint8差1，先校验再计时。
  透明ROI覆盖、legacy_alpha、通道顺序保持既有配置。
  reflection透视采样走原PyTorch fallback；畸变采样仍grid_sample。
  normalization=legacy保留原归一化，matched显式修正采样坐标。
  builder的camera_index_policy=legacy保留原索引，corrected显式修复当前帧行。
  新ROI网格保留局部坐标与z+1e-8，不等同于旧helper的z有效性筛选。
  sigma精确缓存不量化；配置改变后更新gaussian_kernel_cache_version或clear_cache()。
  K/系数/矩阵不按frame_id缓存，当前数值始终作为运行时输入。

生产采样诊断
  from dataclasses import replace
  a = self._paste_batch_adapter
  a.options = replace(a.options, profile_every=30)
  # 预热后，在采样帧之外：
  print(a.profile_summary())
  prepare_cpu_ms为CPU输入准备/提交；core_submit_cpu_ms不是GPU时间。
  export_cpu_ms含等待前序GPU工作；total_ms为同步完整adapter调用。
  CUDA事件区间可能含CPU提交空隙，CPU/CUDA分项不能相加。
  adapter内部采样不含外部prepare_shared_paste_args；完整计时须包含两者。
  正式测速profile_every=0；共享构造对比用新benchmark。
  一个adapter/builder在一个CUDA流上串行使用，跨流输入依赖由调用者建立。

验证与执行图边界
  CPU、Triton interpreter、SM89离线编译检查详见SHARED_PIPELINE_README.md。
  本地无CUDA GPU，未测出新增优化毫秒收益；目标机脚本先校验再计时。
  参数值变化无需逐帧重编译；动态ROI/核尺寸仍影响布局和专用核选择。
  CPU滤波/回调/NumPy输出在图外，本包不是完整CUDA Graph replay方案。
  同相机/设备/分辨率才共享GPU网格；多设备分别构造，CPU滤波结果可共用。
  生产验收使用相同真实帧、参数和负载，对比完整平均/P95及编码/发送流程。
