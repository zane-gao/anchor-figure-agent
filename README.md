# Figure Agent

一个面向论文作者的科研图后编辑 Agent 原型。

当前版本实现了：

- `FigureSceneGraph` 作为统一中间表示
- 双层锚点：语义锚点 + 布局锚点
- 分治式层级结构编辑：全图 / 模块 / 分区 / 区块 / 元素五层
- 栅格初稿到场景图的原型化恢复接口
- 验证层、层级分解、分层协调、布局规划、局部精修、Critic/Stopper 闭环
- PNG / SVG / PPTX 导出
- `AnchorFigureBench-v1` 合成 benchmark 生成
- 自动评测、失败分析、实验 runner 与人工评测模板
- 基准样例生成与最小 Web Demo

## 当前 Idea

这个项目当前的核心想法不是“从论文文本一步生成最终图”，而是先把它做成一个**科研图后编辑 Agent**：

- 输入是 `draft_image + edit_goal`，可选补充 `paper_context / caption / reference_figures`
- 系统先把粗糙栅格初稿恢复为结构化的 `FigureSceneGraph`
- 再围绕 `语义锚点 + 布局锚点` 回答“是什么”和“在哪里”
- 把整图进一步解析为 `全图 -> 大模块 -> 小分区 -> 单元区块 -> 最小元素组件`
- 由分层 agent team 以分治方式处理全局布局、模块组织、局部连线和元素修复
- 最后输出既能直接用于论文、又保留主要组件可编辑性的多格式结果

这里的关键不是把 PNG 一次性修漂亮，而是把**场景图而不是图片本身**作为系统源真相。这样后续的重排、导出、回滚、版本比较和 benchmark 才能稳定成立。

## 总体架构图

```mermaid
flowchart LR
    accTitle: Figure Agent 总体架构
    accDescr: 一个面向科研图后编辑的系统架构图，展示如何从粗糙栅格初稿恢复场景图，经过闭环编辑后导出为可编辑结果。

    draft["输入初稿<br/>draft_image + edit_goal"]
    context["可选上下文<br/>paper_context / caption / reference_figures"]

    subgraph perception["感知与结构恢复"]
        raster["栅格转场景图 Agent<br/>Raster-to-Scene"]
        verify["工具校验层<br/>OCR / 检测 / 分组 / 流程规则"]
        scene["FigureSceneGraph<br/>nodes / edges / groups / constraints"]
        anchors["双层锚点<br/>语义锚点 + 布局锚点"]
        hierarchy["层级分解<br/>global / module / region / block / element"]
    end

    subgraph loop["闭环编辑核心"]
        planner["分层协调器<br/>Global/Module/Region/Block"]
        retouch["元素执行器<br/>局部清理 + 风格规范化"]
        critic["评审与停止器<br/>结构 / 可读性 / 文本 / 可编辑性"]
    end

    subgraph memory["持久状态与记忆"]
        asset["Asset State<br/>场景图 + 渲染快照"]
        execution["Execution State<br/>原子操作序列"]
        planning["Planning State<br/>问题列表 + 目标分数"]
        dag["Version DAG<br/>最佳版本 / 回滚 / 分支"]
    end

    subgraph outputs["最终交付物"]
        png["修复结果图<br/>revised.png"]
        svg["可编辑矢量图<br/>editable.svg"]
        pptx["可编辑演示稿对象<br/>editable.pptx"]
        json["结构与报告<br/>scene_graph.json + edit_report.json"]
    end

    draft --> raster
    context --> raster
    raster --> verify --> scene
    scene --> anchors
    scene --> hierarchy
    anchors --> planner
    hierarchy --> planner
    planner --> retouch --> critic
    critic -->|继续迭代| scene
    critic -->|停止并导出| png
    critic -->|停止并导出| svg
    critic -->|停止并导出| pptx
    critic -->|停止并导出| json

    scene --> asset
    planner --> execution
    critic --> planning
    asset --> dag
    execution --> dag
    planning --> dag

    classDef input fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#082f49
    classDef core fill:#eff6ff,stroke:#2563eb,stroke-width:2px,color:#1e3a8a
    classDef memory fill:#f5f3ff,stroke:#7c3aed,stroke-width:2px,color:#4c1d95
    classDef output fill:#ecfccb,stroke:#65a30d,stroke-width:2px,color:#365314

    class draft,context input
    class raster,verify,scene,anchors,hierarchy,planner,retouch,critic core
    class asset,execution,planning,dag memory
    class png,svg,pptx,json output
```

## 闭环流程图

```mermaid
flowchart TD
    accTitle: 科研图后编辑闭环流程
    accDescr: 一个从粗糙科研图初稿开始，经过结构恢复、布局修复、局部精修和质量评估，最终导出结果的闭环流程图。

    start["开始<br/>粗糙科研图初稿"]
    recover["1. 恢复场景图<br/>从栅格图恢复结构"]
    verify["2. 校验结构<br/>文本 / 分组 / 边 / 阅读顺序"]
    anchor["3. 建立双层锚点<br/>语义绑定 + 布局绑定"]
    layout["4. 结构重排<br/>move / resize / regroup / reroute_arrow"]
    polish["5. 局部精修<br/>replace_text / normalize_style / local_cleanup"]
    judge["6. 质量打分<br/>忠实度 / 可读性 / 可编辑性"]
    stop{"是否停止?"}
    export["导出结果<br/>PNG + SVG + PPTX + JSON report"]
    retry["写入版本 DAG<br/>保留最佳状态并继续迭代"]

    start --> recover --> verify --> anchor --> layout --> polish --> judge --> stop
    stop -->|是| export
    stop -->|否| retry --> layout

    classDef process fill:#f8fafc,stroke:#334155,stroke-width:2px,color:#0f172a
    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#7c2d12
    classDef done fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class start,recover,verify,anchor,layout,polish,judge,retry process
    class stop decision
    class export done
```

## 当前模块映射

- `Raster-to-Scene Agent`、`Tool Verification Layer`、`HierarchicalDecomposer`、`HierarchicalCoordinator`、`LayoutPlanner`、`RetouchExecutor`、`Critic + Stopper` 在 `figure_agent/agents.py`
- `FigureSceneGraph`、锚点、约束、层级结构、版本 DAG、三层记忆状态在 `figure_agent/models.py`
- `PNG / SVG / PPTX` 导出在 `figure_agent/exporters.py`
- `DrawIOAdapter` 预留接口在 `figure_agent/drawio_adapter.py`
- benchmark 样例与粗糙初稿退化构造在 `figure_agent/benchmark.py`
- Web Demo 和 `/api/examples`、`/api/run` 在 `figure_agent/web.py`

## 快速开始

```bash
python main.py benchmark --output-dir runs/AnchorFigureBench-v1 --count-per-family 120
python main.py experiment --config configs/anchorfigure_experiment_main.json
python main.py prepare-human-eval --benchmark-root runs/AnchorFigureBench-v1 --results-root runs/results/main --output-dir runs/human_eval
python main.py run --case-dir runs/AnchorFigureBench-v1/cases/multibranch_method_000_easy --output-dir runs/demo_run
python main.py serve --root-dir runs/AnchorFigureBench-v1 --port 8765
```

打开 `http://127.0.0.1:8765` 可以查看 Demo。

## 目录

- `figure_agent/models.py`: 场景图、锚点、约束、版本 DAG、记忆状态
- `figure_agent/agents.py`: 核心闭环 agent pipeline
- `figure_agent/evaluation.py`: 自动评测指标与失败标签
- `figure_agent/experiments.py`: 主实验 / 基线 / 消融 runner
- `figure_agent/human_eval.py`: 人工评测模板打包
- `figure_agent/exporters.py`: PNG / SVG / PPTX 导出
- `figure_agent/benchmark.py`: `AnchorFigureBench-v1` 数据生成与退化构造
- `figure_agent/web.py`: 简单 Web 服务器与 API
- `configs/`: 主实验与消融实验配置
- `docs/`: problem statement、benchmark schema、执行手册
- `tests/test_pipeline.py`: 最小回归测试

## 原型假设

- v1 只处理方法流程图
- 默认通过 `draft_manifest.json` 模拟 VLM 提议与 OCR / 检测结果
- 对真实图片输入提供基础回退解析，但当前不追求高精度感知
- “可编辑”定义为文字与主要图元可在 PPT / SVG 中单独选中修改
- DrawIO/XML 暂为后续方向，本轮主交付仍为 `SVG/PPTX`
