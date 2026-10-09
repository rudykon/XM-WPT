# XM-WPT：定向聚焦无线传能仿真

固定普通手机可部署的接收面积，利用信标定位与相位调节，研究发射端口径及射频馈入功率对数米距离无线充电的影响。

这是独立研究模型，**不是小米官方代码，也不是小米样机参数或实测结果**。默认条件为 60 GHz、50 cm² 接收面、3 m 距离、144 个调相通道与 5 W 负载目标。模型采用矢量角谱传播、球面相位初始化、局部相位优化，并进行公众环境 E/H 筛查。

## 快速开始

需要 Python 3.12+。在仓库根目录安装依赖并验证保存结果：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run.py test
.venv\Scripts\python run.py --help
```

macOS/Linux 对应解释器路径为 `.venv/bin/python`。以下命令假定已经激活虚拟环境或使用相应解释器：

```text
python run.py analyze
python run.py plot
```

这些命令使用已有结果，不重新运行完整优化；四幅图会生成在仓库内的 `文档/`。

重新计算时依次执行以下命令，会覆盖对应结果和生成文件，且需要较充足内存：

```text
python run.py simulate
python run.py optimize
python run.py analyze
python run.py plot
```

参数设置位于 `wpt/config.py`；已有结果的参数快照位于 `results/focused_aperture/config.json`，后者不是程序输入。修改参数后应重新计算，不能混用不同配置的结果。

## 仓库内容

- `wpt/`：物理模型、优化、分析、绘图与报告生成相关 Python 源码。
- `tests/`：功率守恒、伴随梯度与保存结果回归检查。
- `results/focused_aperture/`：复现分析和绘图所需的 13 个结果快照文件。

保存的默认结果中，30 cm 口径候选需要约 **48.5 W 射频馈入、159 W 墙插输入，墙插效率约 3.15%**。这是有限口径采样与局部优化的结果，不是全局最优；所需场强超过本文设置的公众环境筛查值，不能据此认定开放房间或手持使用可行。

凭据、本地依赖、缓存、历史归档、LaTeX 模板与报告均未上传。报告相关 Python 源码保留，但 `report`、`pdf`、`package` 及包含报告阶段的 `all` 命令需要原本地项目中的 LaTeX 文件；本仓库的复现流程使用上面列出的四个命令。
