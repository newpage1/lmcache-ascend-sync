# lmcache-ascend-sync 工具改进方案

日期：2026-05-06

---

## 1. 当前架构问题

### 1.1 核心缺陷：完整文件生成策略

当前 `PRGenerator` 的工作流程：

```
读取全部 Ascend 源码 + upstream diff
    → 构建巨型 prompt（含所有源文件）
    → LLM 输出完整文件内容
    → write_text() 替换整个文件
```

**问题链条：**

| 环节 | 问题 | 后果 |
|------|------|------|
| Prompt 构建 | 把 1713 行的 `npu_connectors.py` 全文塞入 prompt | LLM 上下文窗口被占满 |
| LLM 生成 | 倾向于"重写"而非"修改" | 大量保留逻辑被丢弃 |
| 文件写入 | `write_text()` 替换整个文件 | 无 diff、无回滚、无检查点 |
| 验证 | 无任何验证机制 | 74% 代码丢失未被发现 |

### 1.2 PR #227 实测证据

v0.4.3→v0.4.4 upstream 实际只需改 **3-5 行**，但 LLM 生成的结果：

| 文件 | 原始行数 | 生成行数 | 丢失率 | 评价 |
|------|---------|---------|--------|------|
| `__init__.py` | ~400 | ~400 | 0% | 版本号正确，漏掉 config 变更 |
| `cache_engine.py` | 不存在 | 81 | N/A | 不该生成（在未合并 PR #221 中） |
| `npu_connectors.py` | 1713 | 443 | **74%** | 核心功能全部丢失 |
| `storage_backend/__init__.py` | ~200 | ~200 | 0% | 不需要修改 |

丢失的关键功能：
- `VLLMBufferLayerwiseNPUConnector`: `_prepare_transfer_context`、完整 `batched_to_gpu`/`batched_from_gpu` 生成器
- `VLLMPagedMemNPUConnectorV2`: `from_metadata` 工厂方法、310P 适配、P2P pipeline
- `SGLangLayerwiseNPUConnector`: 全部实现变成 `pass`

### 1.3 Prompt 设计问题

```
当前 prompt 模板（_build_prompt）:
├── "Per-File Action Plan" → LLM 以为每个文件需要大幅修改
├── "Output the COMPLETE file content" → 强制完整文件输出
├── 缺乏约束 "不能删除现有方法" → LLM 自由删减
└── 发送全部源码 → 噪声太多，信号被淹没
```

### 1.4 缺乏变更分类

所有 upstream 变更被一视同仁地交给 LLM 处理，但实际上：

| 变更类型 | 占比(v0.4.4) | 是否需要 LLM |
|----------|-------------|-------------|
| 标识符重命名（`is_tuple_format` → `is_separate_format`） | 30% | 否，字符串替换即可 |
| import 去条件化（去掉 `if torch.cuda.is_available()`） | 20% | 否，确定性删除即可 |
| 新增 config 字段 | 20% | 否，模板注入即可 |
| 新增方法签名 | 10% | 部分，需要 LLM 适配 Ascend |
| 复杂逻辑修改 | 20% | 是，需要 LLM 理解语义 |

**结论：60-70% 的 upstream 变更可以用确定性规则处理，不需要 LLM。**

---

## 2. 新架构设计

### 2.1 总体架构

```
                    ┌──────────────────────────┐
                    │  Phase 1: Upstream Diff   │
                    │  Analysis (现有)          │
                    │  ConflictAnalyzer         │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 2: Change          │
                    │  Classification           │  (新增)
                    │  变更分类器               │
                    └────────────┬─────────────┘
                                 │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
      ┌────────▼────────┐ ┌─────▼──────┐ ┌────────▼────────┐
      │  Deterministic  │ │  Config     │ │  Targeted LLM   │
      │  Transform      │ │  Injector   │ │  (Method-level)  │
      │                 │ │             │ │                  │
      │  确定性字符串   │ │  配置字段   │ │  只输出修改的    │
      │  替换规则       │ │  注入       │ │  方法体          │
      └────────┬────────┘ └─────┬──────┘ └────────┬────────┘
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 3: Verification    │  (新增)
                    │  变更质量验证             │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Phase 4: PR Creation     │
                    │  (现有，增强)             │
                    └──────────────────────────┘
```

### 2.2 新增模块

```
src/
├── pr_generator.py          # 现有，重构
├── change_classifier.py     # 新增：变更分类器
├── deterministic_transform.py # 新增：确定性变换引擎
├── method_patcher.py         # 新增：方法级 LLM patch
├── verifier.py               # 新增：变更质量验证
└── transform_rules.yaml      # 新增：变换规则配置
```

---

## 3. 各模块详细设计

### 3.1 `change_classifier.py` — 变更分类器

**职责：** 解析 upstream diff，将每个 hunk 归类为确定性变换或需要 LLM 处理。

```python
class ChangeClassifier:
    """将 upstream diff hunks 分类为不同处理策略。"""

    # 分类枚举
    MECHANICAL_RENAME = "mechanical_rename"     # 标识符改名
    IMPORT_CHANGE = "import_change"             # import 语句变化
    CONFIG_ADDITION = "config_addition"         # 新增配置字段
    PARAM_CHANGE = "param_change"               # 方法签名变化
    LOGIC_CHANGE = "logic_change"               # 复杂逻辑修改
    NEW_METHOD = "new_method"                   # 新增方法/类
    IRRELEVANT = "irrelevant"                   # 与 Ascend 无关

    def classify_diff(self, diff_text: str, patch_points: dict) -> list[ClassifiedChange]:
        """返回分类后的变更列表，每个变更附带处理策略。"""
        ...

    def _is_relevant_file(self, filepath: str, patch_points: dict) -> bool:
        """检查 diff 文件是否在 patch_points 关注列表中。"""
        ...

    def _classify_hunk(self, hunk: DiffHunk) -> str:
        """对单个 diff hunk 分类。"""
        # 启发式规则：
        # - 只改标识符名称 → MECHANICAL_RENAME
        # - 只改 import 行 → IMPORT_CHANGE
        # - 在 config 字典中新增键值 → CONFIG_ADDITION
        # - 方法签名新增/删除参数 → PARAM_CHANGE
        # - 函数体内部逻辑变化 → LOGIC_CHANGE
        # - 新增 def/class → NEW_METHOD
        ...
```

**输入：** `git diff v0.4.3..v0.4.4` 的文本
**输出：** 分类后的变更列表，每项包含 `file, hunk, category, strategy`

### 3.2 `deterministic_transform.py` — 确定性变换引擎

**职责：** 用确定性规则处理不需要 LLM 的变更。

```python
class DeterministicTransformer:
    """执行确定性代码变换。"""

    def __init__(self, rules_path: str = "config/transform_rules.yaml"):
        self.rules = self._load_rules(rules_path)

    def transform(self, ascend_sources: dict[str, str],
                  classified_changes: list[ClassifiedChange]) -> dict[str, str]:
        """对所有确定性变更执行变换，返回修改后的文件内容。"""
        ...

    def _apply_rename(self, source: str, change: ClassifiedChange) -> str:
        """标识符重命名：精确的字符串替换。"""
        ...

    def _apply_import_change(self, source: str, change: ClassifiedChange) -> str:
        """import 变更：删除条件守卫、更新路径。"""
        ...

    def _apply_config_injection(self, source: str, change: ClassifiedChange) -> str:
        """配置注入：在 _CONFIG_DEFINITIONS 中添加新字段。"""
        ...
```

### 3.3 `transform_rules.yaml` — 变换规则

```yaml
# 版本特定的变换规则
# 每次适配完成后，成功的规则保留为历史参考

rules:
  # 通用规则（跨版本适用）
  - name: "cuda_to_npu"
    type: "replace"
    pattern: "torch.cuda"
    replacement: "torch.npu"
    scope: "ascend_only"  # 只应用于 Ascend 文件
    description: "CUDA → NPU 设备 API 替换"

  - name: "assert_vllm_flash_attn"
    type: "rename"
    pattern: "assert_is_vllm_flash_attn_or_flash_infer"
    replacement: "assert_is_vllm_mla_or_flash_attn_or_flash_infer"
    files:
      - "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    description: "v0.4.4: 断言函数增加 MLA 支持"

  - name: "import_unguard_c_ops"
    type: "import_unguard"
    pattern: 'if torch\\.cuda\\.is_available\\(\\):\\s+import lmcache\\.c_ops'
    replacement: "import lmcache.c_ops"
    files:
      - "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    description: "v0.4.4: 去掉 c_ops import 的 CUDA 条件守卫"

  - name: "memory_format_mla"
    type: "conditional_replace"
    pattern: "MemoryFormat\\.KV_T2D"
    replacement: "MemoryFormat.KV_MLA_FMT if self.use_mla else MemoryFormat.KV_T2D"
    files:
      - "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    context_pattern: "def.*buffer.*alloc|def.*prepare"
    description: "v0.4.4: MemoryFormat 根据 use_mla 动态选择"

  - name: "config_gds_rename"
    type: "config_rename"
    old_field: "cufile_buffer_size"
    new_field: "gds_buffer_size"
    target_file: "lmcache_ascend/__init__.py"
    description: "v0.4.4: cufile_buffer_size → gds_buffer_size"

  - name: "config_new_fields"
    type: "config_add"
    fields:
      - name: "local_disk_path_sharding"
        type: "str"
        default: '"by_gpu"'
      - name: "pd_skip_proxy_notification"
        type: "bool"
        default: "False"
    target_file: "lmcache_ascend/__init__.py"
    description: "v0.4.4: 新增配置字段"
```

### 3.4 `method_patcher.py` — 方法级 LLM Patch

**职责：** 仅对需要 LLM 的复杂变更，做外科手术式的方法级替换。

```python
class MethodPatcher:
    """使用 LLM 对单个方法进行精确修改。"""

    def __init__(self, llm_client, model: str, max_tokens: int):
        self.client = llm_client
        self.model = model
        self.max_tokens = max_tokens

    def patch_methods(self, ascend_sources: dict[str, str],
                      changes_needing_llm: list[ClassifiedChange],
                      upstream_before: dict[str, str],
                      upstream_after: dict[str, str]) -> dict[str, str]:
        """对所有需要 LLM 的变更，逐方法处理。"""
        ...

    def _extract_method(self, source: str, class_name: str, method_name: str) -> str:
        """用 AST 提取单个方法的源码。"""
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            return ast.get_source_segment(source, item)
        return None

    def _replace_method(self, source: str, class_name: str,
                        method_name: str, new_body: str) -> str:
        """用 AST 精确替换单个方法。"""
        ...

    def _build_method_prompt(self, method_name: str,
                             upstream_before: str,
                             upstream_after: str,
                             ascend_version: str) -> str:
        """构建精简的方法级 prompt。

        关键：只发送变更的方法，不发送整个文件。
        prompt 大小控制在 ~1000 token 以内。
        """
        return f"""Upstream changed method `{method_name}`:

--- before (v0.4.3)
{upstream_before}

+++ after (v0.4.4)
{upstream_after}

Current Ascend version of `{method_name}`:
{ascend_version}

Apply the equivalent change to the Ascend version.
Rules:
- Keep ALL Ascend-specific logic (torch.npu, lmc_ops, is_310p, etc.)
- Only modify what upstream changed
- Output ONLY the modified method body (including `def` line)
- Do NOT output anything else
"""
```

**与当前方案的核心区别：**

| 维度 | 当前方案 | 方法级 Patch |
|------|---------|-------------|
| LLM 输入 | 整个项目源码 (~20K token) | 单个方法 (~500 token) |
| LLM 输出 | 完整文件 (~5K token) | 修改后的方法 (~200 token) |
| 替换方式 | `write_text()` 覆盖整个文件 | AST 定位 + 替换单个方法 |
| 风险范围 | 整个文件 | 单个方法 |
| 可回滚性 | 差（只能回滚整个文件） | 好（每个方法独立 patch） |

### 3.5 `verifier.py` — 变更质量验证

**职责：** 在 PR 创建前自动验证生成代码的质量。

```python
class ChangeVerifier:
    """验证生成代码的质量。"""

    def verify(self, original_sources: dict[str, str],
               modified_sources: dict[str, str],
               classified_changes: list[ClassifiedChange]) -> VerificationReport:
        """运行所有检查，返回验证报告。"""
        results = []
        for filepath, new_content in modified_sources.items():
            original = original_sources.get(filepath, "")
            file_result = self._verify_file(filepath, original, new_content)
            results.append(file_result)
        return VerificationReport(results=results, passed=all(r.passed for r in results))

    def _verify_file(self, filepath: str, original: str, modified: str) -> FileVerification:
        """对单个文件执行所有检查。"""
        checks = [
            self._check_syntax(modified),
            self._check_line_count_delta(filepath, original, modified),
            self._check_class_preservation(original, modified),
            self._check_method_preservation(original, modified),
            self._check_import_preservation(original, modified),
            self._check_ascend_keywords(modified),
        ]
        ...

    # --- 具体检查 ---

    def _check_syntax(self, code: str) -> CheckResult:
        """py_compile 检查语法。"""
        import py_compile
        import tempfile
        ...

    def _check_line_count_delta(self, filepath: str,
                                 original: str, modified: str) -> CheckResult:
        """行数偏差检查：单文件变更不超过 ±30%。"""
        if not original:
            return CheckResult("line_count", "passed", "New file")
        orig_lines = len(original.strip().split("\n"))
        new_lines = len(modified.strip().split("\n"))
        delta_pct = abs(new_lines - orig_lines) / orig_lines * 100
        if delta_pct > 30:
            return CheckResult("line_count", "FAILED",
                f"{delta_pct:.0f}% line count change (>{30}% threshold)")
        ...

    def _check_class_preservation(self, original: str, modified: str) -> CheckResult:
        """AST 检查：所有原有 class 仍存在。"""
        import ast
        orig_classes = {n.name for n in ast.walk(ast.parse(original))
                        if isinstance(n, ast.ClassDef)}
        new_classes = {n.name for n in ast.walk(ast.parse(modified))
                       if isinstance(n, ast.ClassDef)}
        missing = orig_classes - new_classes
        if missing:
            return CheckResult("class_preservation", "FAILED",
                f"Missing classes: {missing}")
        ...

    def _check_method_preservation(self, original: str, modified: str) -> CheckResult:
        """AST 检查：所有原有 public method 仍存在。"""
        import ast
        # 提取所有 class 的 public method 名称
        orig_methods = self._get_public_methods(original)
        new_methods = self._get_public_methods(modified)
        missing = orig_methods - new_methods
        if missing:
            return CheckResult("method_preservation", "FAILED",
                f"Missing methods: {missing}")
        ...

    def _check_ascend_keywords(self, code: str) -> CheckResult:
        """检查 Ascend 特有关键词未被丢失。"""
        keywords = ["torch.npu", "lmc_ops", "is_310p", "Ascend",
                     "npu", "NPU", "CANN"]
        missing = [kw for kw in keywords if kw in code]
        # 注意：不是所有文件都包含所有关键词
        # 这个检查应该与原始文件对比，确保没有丢失已有的关键词
        ...
```

**验证报告格式：**

```
验证报告: v0.4.4 adaptation
==================================================
文件: lmcache_ascend/v1/npu_connector/npu_connectors.py
  [PASSED] 语法检查
  [FAILED] 行数偏差: 74% (阈值 30%)
  [FAILED] 类完整性: 缺少 VLLMPagedMemNPUConnectorV2
  [FAILED] 方法完整性: 缺少 from_metadata, to_gpu_310p, ...
  [PASSED] import 完整性
  [FAILED] Ascend 关键词: 丢失 torch.npu (出现 5→2 次)
==================================================
总结果: FAILED — 拒绝合并
```

---

## 4. 重构后的 `PRGenerator` 主流程

```python
class PRGenerator:
    def generate(self, from_version, to_version, analysis, rebase_result):
        # Phase 1: 已有 — ConflictAnalyzer 结果
        upstream_diff = analysis.get("diff_summary", "")
        ascend_sources = self._read_ascend_sources(downstream_path)

        # Phase 2: 变更分类 (新增)
        classifier = ChangeClassifier()
        classified = classifier.classify_diff(upstream_diff, self.patch_points)

        # Phase 3a: 确定性变换 (新增)
        transformer = DeterministicTransformer()
        deterministic_changes = [c for c in classified
                                  if c.strategy in ("mechanical_rename",
                                                     "import_change",
                                                     "config_addition")]
        modified_sources = transformer.transform(ascend_sources, deterministic_changes)

        # Phase 3b: 方法级 LLM patch (新增)
        llm_changes = [c for c in classified
                        if c.strategy in ("logic_change", "param_change", "new_method")]
        if llm_changes:
            patcher = MethodPatcher(self.client, self.model, self.max_tokens)
            llm_modified = patcher.patch_methods(
                modified_sources, llm_changes,
                upstream_before, upstream_after,
            )
            modified_sources.update(llm_modified)

        # Phase 4: 验证 (新增)
        verifier = ChangeVerifier()
        report = verifier.verify(ascend_sources, modified_sources, classified)
        if not report.passed:
            # 保存生成代码 + 验证报告，不创建 PR
            self._save_with_report(to_version, modified_sources, report)
            return {"success": False,
                    "error": f"Verification failed:\n{report.summary()}"}

        # Phase 5: 创建 PR (现有逻辑)
        return self._create_pr(to_version, modified_sources, analysis, rebase_result)
```

---

## 5. 实施计划

### Phase A：防御层（P0，防止灾难）

**目标：** 在当前架构上加验证，防止再次出现 74% 代码丢失。

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 实现 `verifier.py` | `src/verifier.py` | ~150 行 |
| 在 `_create_pr` 前调用验证 | `src/pr_generator.py` | ~20 行修改 |
| 验证失败时保存代码+报告 | `src/pr_generator.py` | ~30 行修改 |

**验证规则（MVP）：**
1. 语法检查（`py_compile`）
2. 行数偏差检查（单文件 >30% 则 FAILED）
3. 类完整性检查（AST 解析）
4. 方法完整性检查（AST 解析）

### Phase B：确定性变换（P1，减少 LLM 依赖）

**目标：** 60-70% 的变更用确定性规则处理，不经过 LLM。

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 实现 `change_classifier.py` | `src/change_classifier.py` | ~200 行 |
| 实现 `deterministic_transform.py` | `src/deterministic_transform.py` | ~250 行 |
| 创建 `transform_rules.yaml` | `config/transform_rules.yaml` | ~80 行 |
| 重构 `PRGenerator.generate()` | `src/pr_generator.py` | ~100 行修改 |

**支持的变换类型：**
- `replace` — 字符串替换（精确匹配）
- `rename` — 标识符重命名（精确匹配）
- `import_unguard` — 删除 import 条件守卫
- `config_add` — 注入新配置字段到 `_CONFIG_DEFINITIONS`
- `config_rename` — 配置字段重命名
- `conditional_replace` — 带上下文条件的替换

### Phase C：方法级 LLM Patch（P1，提升 LLM 质量）

**目标：** LLM 只输出修改的方法，用 AST 精确替换。

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 实现 `method_patcher.py` | `src/method_patcher.py` | ~300 行 |
| AST 方法提取/替换工具 | `src/method_patcher.py` | ~150 行 |
| 方法级 prompt 模板 | `src/method_patcher.py` | ~50 行 |
| 重构 `PRGenerator.generate()` | `src/pr_generator.py` | ~50 行修改 |

**关键实现细节：**
- 用 `ast.get_source_segment()` 提取方法源码
- 用 `ast.parse()` + `ast.fix_missing_locations()` 替换方法节点
- Prompt 大小控制在 ~1000 token（当前 ~20K token）
- LLM 输出大小控制在 ~500 token（当前 ~5K token）

### Phase D：增强功能（P2）

| 任务 | 描述 |
|------|------|
| Dry-run 模式 | 生成代码但不 push，保存到本地供人工审查 |
| 变更规则积累 | 每次成功适配后，将规则保存到规则库 |
| PR 评论 | 验证报告作为 PR comment 发布 |
| 回滚机制 | 每个方法独立 patch，支持逐方法回滚 |
| 跨版本规则复用 | 分析多个版本的 diff，提取通用变换模式 |

---

## 6. 预期效果

| 指标 | 当前 | Phase A 后 | Phase B+C 后 |
|------|------|-----------|-------------|
| 代码丢失率 | 74% | <5%（验证拦截） | <2%（确定性+方法级） |
| LLM prompt 大小 | ~20K token | ~20K token（不变） | ~1K token（方法级） |
| LLM 输出大小 | ~5K token | ~5K token（不变） | ~500 token（方法级） |
| API 调用次数 | 1 次 | 1 次 | 1-5 次（每个方法独立） |
| 确定性变更准确率 | N/A | N/A | ~100% |
| 整体适配准确率 | ~30% | ~30%（但会被验证拦截） | >90% |
| 可回滚性 | 差（整个文件） | 差（整个文件） | 好（逐方法） |
| 失败可见性 | 低（只能看 PR diff） | 高（验证报告） | 高（验证报告） |

---

## 7. 风险和注意事项

### 7.1 AST 解析的局限性

- `ast.get_source_segment()` 需要精确的缩进信息
- 装饰器（decorator）需要特殊处理
- 类变量（class variable）和方法（method）需要区分
- **缓解：** 充分的单元测试，覆盖各种 Python 语法模式

### 7.2 变更分类的准确性

- 启发式规则可能有误判（把 logic_change 误判为 param_change）
- **缓解：** 分类结果可供人工 review；对于模糊情况，默认走 LLM 路径

### 7.3 确定性规则的维护

- 每个版本可能有不同的变换规则
- **缓解：** 规则以 YAML 配置文件维护，不硬编码；积累跨版本通用规则

### 7.4 方法级 LLM 的边界情况

- 跨方法的重构（如方法合并、拆分）无法用方法级 patch 处理
- **缓解：** 分类器检测到跨方法变更时，退回到文件级 LLM（但仍有验证层保护）

---

## 8. 对 v0.4.4 的建议

鉴于工具改进尚需时间，v0.4.4 适配建议手动完成（预计 ~30 分钟）：

1. `__init__.py`：更新 `LMCACHE_UPSTREAM_TAG`，添加新 config 字段，处理 `cufile`→`gds` 重命名
2. `npu_connectors.py`：3 处确定性修改：
   - `is_tuple_format()` → `is_separate_format()`
   - 断言函数名加 MLA
   - `MemoryFormat.KV_T2D` 改为根据 `use_mla` 动态选择
3. 不需要新建文件
4. 不需要修改 `storage_backend/__init__.py`

**关闭 PR #227**，手动适配后创建新 PR。
