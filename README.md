# Spice

[官网](https://spiceagent.vercel.app/) · [GitHub](https://github.com/beelovelife/spice)

> 一个使用 Python 构建的本地 coding agent，提供 CLI / TUI 两种交互形态，支持多 LLM provider、可扩展工具体系、会话持久化、上下文压缩、技能（skills）与本地扩展（extensions）。

Spice 强调架构边界清晰、运行链路可追踪、会话可恢复、工具受控。整体心态是「核心稳、能力放边缘」：默认尽量从 CLI 命令、skill、扩展、配置项接入新能力，避免把 agent loop 写成万能大文件。

---

## ✨ 主要特性

- **多 provider 支持**：内置 OpenAI、DeepSeek、Anthropic、Gemini 等模型，包括 Claude Haiku 4.5、Sonnet 4.6 与 Opus 4.8，统一通过模型注册表与 `~/.spice/settings.json` 管理。
- **双前端体验**：`spice` / `spice chat` 走基于 `prompt_toolkit` 的交互式 CLI，`spice tui` 进入全屏 TUI；二者复用同一套 agent、session 服务和 command registry。
- **流式 agent loop**：明确的 `prompt → turn_start → assistant stream → tool calls → tool results → turn_end → agent_end` 事件流，UI、会话持久化、trace 都挂在事件上。
- **受控工具体系**：内置 `file`、`bash`、`web`、`skill`、`memory`、`subagent` 等工具集合，统一 schema / 执行 / 错误结构，支持 read-only 模式与确认门。
- **会话持久化与树状历史**：默认使用可读的 JSONL 文件，也可切换 SQLite；支持 `resume`、`rewind`、`prune`、`workspaces` 等管理命令。
- **受控执行环境**：默认在本机 workspace 策略下执行，也可使用 Docker sandbox 隔离命令运行。
- **上下文压缩**：手动 `/compact` 与自动 compaction，配合可序列化上下文与摘要原因。
- **Skills 与 Extensions**：`~/.spice/skills/` 沉淀工作流，`~/.spice/extensions/*.py` 提供可信本地扩展，能注册工具、slash command 与 hook。
- **MCP 客户端**：通过 stdio 或 Streamable HTTP 连接 MCP Server，将发现的工具接入统一 Tool、确认、trace 与超时链路。
- **长任务与子代理**：内置 long task 状态机和 `spawn_subagents` 工具，可在主 agent 中分发子任务。
- **诊断友好**：`spice logs`、`spice config`、`spice skills doctor`、`spice memory status` 等命令把运行状态与配置完全可视化。

---

## 🚀 快速开始

### 1. 环境要求

- Python `>= 3.11`
- [`uv`](https://docs.astral.sh/uv/) 管理安装、依赖与虚拟环境
- Git（从 GitHub 安装或参与开发时需要）
- macOS / Linux（Windows 未做覆盖测试）

```bash
uv --version
python3 --version
```

尚未安装 uv 时：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装 Spice

推荐安装为隔离的全局工具，之后可以在任意工作目录直接运行 `spice`：

```bash
uv tool install git+https://github.com/beelovelife/spice.git
spice --version
```

需要查看或修改源码时：

```bash
git clone https://github.com/beelovelife/spice.git
cd spice
uv sync
source .venv/bin/activate
spice --version
```

#### 更新 Spice

通过 `uv tool` 安装后，使用以下命令升级到最新版本：

```bash
uv tool upgrade spice
```

如果升级后仍未拉取到最新代码，可以绕过 uv 本地缓存，强制从 GitHub 重新安装：

```bash
uv tool install --force git+https://github.com/beelovelife/spice.git
```

### 3. 配置 API Key

首次使用可先配置 OpenAI 或 DeepSeek。Spice 优先读取环境变量，未命中时读取 `~/.spice/secrets.json`：

```bash
export OPENAI_API_KEY="sk-..."
# 或
export DEEPSEEK_API_KEY="sk-..."
```

也可以创建私有密钥文件：

```json
{
  "OPENAI_API_KEY": "sk-...",
  "DEEPSEEK_API_KEY": "sk-..."
}
```

支持的密钥如下：

| Provider  | 环境变量                                  |
|-----------|-------------------------------------------|
| openai    | `OPENAI_API_KEY`                          |
| deepseek  | `DEEPSEEK_API_KEY`                        |
| anthropic | `ANTHROPIC_API_KEY`                       |
| gemini    | `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`      |
| tavily    | `TAVILY_API_KEY`                          |

### 4. 选择模型

进入交互式 CLI 后运行 `/models`，或通过命令设置默认模型：

```bash
spice models
spice config set default-model gpt-5.1
```

也可以在 `~/.spice/settings.json` 中定义 profile。API key 不应写入该文件：

```json
{
  "defaultModel": "deepseek-v4-pro",
  "models": {
    "deepseek-v4-pro": {
      "provider": "deepseek",
      "model": "deepseek-v4-pro",
      "baseUrl": "https://api.deepseek.com",
      "protocol": "openai-completions",
      "temperature": 0.2
    }
  }
}
```

### 5. 运行

```bash
spice                           # 交互式 CLI
spice tui                       # 全屏 TUI
spice run "你好，介绍一下自己" # 单次 prompt
```

从未激活的源码环境运行时，在上述命令前加 `uv run`：

```bash
uv run spice
uv run spice tui
uv run spice run "你好，介绍一下自己"
```

---

## 🧭 命令一览

### 顶层命令

| 命令                                | 说明                                                |
|-------------------------------------|-----------------------------------------------------|
| `spice`                             | 进入交互式 CLI（默认入口）                          |
| `spice chat`                        | 显式进入交互式 CLI                                  |
| `spice tui`                         | 进入全屏 TUI                                        |
| `spice chat --trace`                | 交互式 CLI，并记录本次启动的多轮 trace              |
| `spice tui --trace`                 | 全屏 TUI，并记录本次启动的多轮 trace                |
| `spice run "<prompt>"`              | 单次执行 prompt，stdout 保持脚本化                  |
| `spice run "<prompt>" --trace`      | 单次执行并写入结构化 runtime trace                  |
| `spice trace inspect <path.json>`    | 查看 trace 的步骤、工具结果、usage 和事件           |
| `spice resume <session-id>`         | 按 id 恢复一个会话                                  |
| `spice models`                      | 列出 provider/model 与当前配置、API key 状态        |
| `spice logs [--tail N] [--path]`    | 查看 Spice 运行日志路径与最近日志                   |
| `spice --version` / `-v`            | 显示版本                                            |
| `spice --debug`                     | 开启调试日志                                        |

### 会话管理（`spice sessions`）

| 子命令                                              | 说明                                                |
|-----------------------------------------------------|-----------------------------------------------------|
| `spice sessions [--limit N] [--all]`                | 列出当前 cwd 的会话（可跨 cwd）                     |
| `spice sessions show <id> [--tree] [--raw]`         | 查看会话历史或树状结构                              |
| `spice sessions rewind <id> <entry-id>`             | 把当前 leaf 指针移到某个 entry                      |
| `spice sessions stats [--all]`                      | 会话数量、空会话数、存储路径等指标                  |
| `spice sessions workspaces`                         | 列出有过会话的所有工作目录                          |
| `spice sessions delete <id> [--yes]`                | 删除单个会话                                        |
| `spice sessions prune --keep-recent N \| --before … \| --from … --to … \| --all-sessions [--yes]` | 按规则批量清理 |

### 配置（`spice config`）

| 子命令                              | 说明                                                |
|-------------------------------------|-----------------------------------------------------|
| `spice config show`                 | 打印当前配置和 settings/secrets 路径                |
| `spice config path`                 | 仅打印配置文件路径                                  |
| `spice config get <key>`            | 读取某个配置项                                      |
| `spice config set <key> <value>`    | 设置模型、密钥、memory、storage、工具运行参数和模型路由 |

支持的配置键包括：

- `default-model`、`api-key`
- `memory.enabled`、`logging.retention_days`
- `storage.backend`、`storage.sqlitePath`
- `tools.max_concurrency`、`tools.default_timeout_seconds`
- `modelRouting.retry.enabled`、`modelRouting.retry.maxAttempts`
- `modelRouting.fallback.enabled`、`modelRouting.fallback.profiles`

`--debug` 提升标准运行日志级别；`run --trace` 生成结构化运行轨迹。

交互执行过程中，TUI 可用 `Esc` 或 `Ctrl+C` 取消当前模型/工具调用并继续会话；普通 CLI 可用 `Ctrl+C`。取消不会回滚工具已经产生的文件或外部副作用。

### Sandbox（`spice sandbox`）

| 子命令                              | 说明                                                |
|-------------------------------------|-----------------------------------------------------|
| `spice sandbox status`              | 查看当前 workspace 的执行模式与后端状态             |
| `spice sandbox init`                | 创建、启动并检查 Docker sandbox；本机模式直接 ready |
| `spice sandbox exec "<command>"`    | 通过当前 sandbox 后端执行 shell 命令                 |
| `spice sandbox stop`                | 停止 Docker 容器；workspace/local 模式不支持 stop   |

### Storage（`spice storage`）

默认使用 `~/.spice/sessions/` 下的 JSONL 文件。切换 SQLite 时：

```bash
spice config set storage.backend sqlite
spice config set storage.sqlitePath ~/.spice/spice.db
spice storage init
```

### 技能（`spice skills`）

| 子命令                                  | 说明                                                |
|-----------------------------------------|-----------------------------------------------------|
| `spice skills list`                     | 列出当前可用 skill（来源、优先级、触发词）          |
| `spice skills view <name> [-f file]`    | 查看 SKILL.md 或 skill 目录下的某个文件             |
| `spice skills doctor`                   | 显示 skill 加载诊断（冲突、解析失败等）             |

### MCP（`spice mcp`）

Spice 从 `~/.spice/settings.json` 的 `mcpServers` 读取全局服务器，并兼容项目级 `.mcp.json`。项目配置的 stdio 命令首次启动前会请求信任确认。

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "remote": {
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer ${REMOTE_MCP_TOKEN}"}
    }
  }
}
```

密钥应放在环境变量或 `~/.spice/secrets.json`，配置中通过 `${VAR}` 引用。

| 命令 | 说明 |
|---|---|
| `spice mcp list` | 列出合并后的 MCP 配置 |
| `spice mcp add <name> --command ... --arg ...` | 添加全局 stdio Server |
| `spice mcp add <name> --url ...` | 添加全局 Streamable HTTP Server |
| `spice mcp test <name>` | 连接并检查 Server |
| `spice mcp tools <name>` | 查看发现的工具及确认策略 |
| `spice mcp enable\|disable <name>` | 启用或禁用全局 Server |
| `spice mcp remove <name>` | 删除全局 Server |
| `spice mcp trust <name>` | 信任当前项目的 stdio Server 配置 |

交互模式使用 `/mcp` 查看连接状态，使用 `/mcp reload` 重载配置和工具。

### 长期记忆（`spice memory`）

| 子命令                  | 说明                                                |
|-------------------------|-----------------------------------------------------|
| `spice memory status`   | 显示记忆历史、游标、容量等状态                      |
| `spice memory enable`   | 开启长期记忆                                        |
| `spice memory disable`  | 关闭长期记忆                                        |
| `spice memory distill`  | 触发一次记忆萃取（把会话总结写入 MEMORY.md）        |

---

## 💬 交互式 Slash Commands

进入 `spice` / `spice chat` / `spice tui` 后，输入 `/` 触发命令补全：

| 命令                | 说明                                                |
|---------------------|-----------------------------------------------------|
| `/models`           | 选择 provider/model，并更新当前会话                 |
| `/sessions`         | 浏览当前 cwd 的会话列表                             |
| `/resume`           | 打开会话选择器，恢复某个历史会话                    |
| `/clear`            | 清空可见对话（不删除会话）                          |
| `/reset`            | 确认后清空当前会话所有消息                          |
| `/delete [id|current]` | 删除会话（默认 current）                         |
| `/history [--tree|--raw]` | 查看当前会话历史                              |
| `/rewind <entry-id>`| 把当前 leaf 指针移到某个 entry                      |
| `/tools`            | 查看内置工具与启用状态                              |
| `/mcp [reload]`     | 查看 MCP 连接状态，或重载服务器和工具               |
| `/settings`         | 当前交互设置（模型、reasoning、工具开关、输出模式） |
| `/cost`、`/usage`   | 查看当前会话的 token usage 与预估费用               |
| `/subagent [on|off|status]` | 控制子代理工具开关                          |
| `/compact [status|focus]`  | 手动压缩上下文                              |
| `/memory [project|global|all|status|show ...]` | 萃取或查看长期记忆       |
| `/plan [task|execute|cancel]` | 进入只读 plan 模式或规划任务             |
| `/task <objective|status|cancel|complete>` | 长任务管理                     |
| `/goal <objective|status|cancel|complete>` | 长期目标管理                   |
| `/skills`           | 列出已安装 skill                                    |
| `/skill:<name>`     | 直接调用某个 skill                                  |
| `/help`             | 显示所有 slash command                              |
| `/quit`             | 退出                                                |

CLI 与 TUI 复用 command registry 和会话服务；少量界面相关的选择与渲染逻辑由各自前端实现。

---

## 🧩 内置工具集

工具按集合（toolset）注册，可通过 `/tools` 查看启用状态：

| Toolset    | 工具                                                                                          |
|------------|-----------------------------------------------------------------------------------------------|
| `core`     | `get_current_time`                                                                            |
| `file`     | `list_dir`、`read_file`、`read_files`、`write_file`、`edit_file`、`apply_patch`、`search_files` |
| `bash`     | `bash`                                                                                        |
| `web`      | `web_search`（依赖 Tavily）                                                                   |
| `skill`    | `skills_list`、`skill_view`                                                                   |
| `memory`   | `memory`                                                                                      |
| `subagent` | `spawn_subagents`                                                                             |

所有工具走统一的 `Tool` 协议（`schema` + `execute`），错误结构统一为结构化 `tool_error`。read-only 集合（如 `list_dir`、`read_file`、`web_search`、`skill_view` 等）默认无需确认门，写操作类工具（`write_file`、`edit_file`、`apply_patch`、`bash`）会触发确认提示。

---

## 🪄 Skills

Skill 是 Markdown 描述的复用工作流。Spice 从以下位置加载 `<name>/SKILL.md`，同名时后列来源覆盖前列来源：

1. 用户级：`~/.spice/skills/<name>/SKILL.md`
2. 项目级：从当前目录向上查找最近的 `.spice/skills/<name>/SKILL.md`
3. 显式路径：通过 `--skill-path` 或 API 提供的 skill 文件/目录

每个 skill 支持 `description`、`triggers`、`always` 等 frontmatter，可被模型通过 `skills_list` / `skill_view` 工具检索查看，或通过 `/skill:<name>` 直接触发。

---

## 🔌 Extensions

Extension 使用可信本地 Python 代码，扫描路径：

- `~/.spice/extensions/*.py`
- `~/.spice/extensions/<name>/{__init__.py,extension.py,main.py,index.py}`

Extension 入口暴露 `extension(api)` / `activate(api)` / `default(api)`，可：

- `@api.tool(...)` 装饰函数并注册模型可调用工具（参数 schema 可从类型注解推导，也可显式传入）
- `api.command(...)` 注册 slash command
- `api.on(...)` 注册 hook（`input`、`tool_call_start`、`tool_call_end` 等）

Agent runtime 事件（`agent_start`、`turn_start`、`text_delta`、`assistant_message`、`tool_execution_start/end`、`agent_end`、`agent_error`）会转发给 extension observer hook。Extension 与 Spice 进程同权限运行；后续若加载项目内 extension，会先补 project trust。

---

## 🗂️ 项目结构

```text
spice/
├── agent/        agent loop、事件、会话内状态、compaction、plan/long task、subagent
├── cli/          typer 命令、prompt_toolkit 交互、rich 渲染
├── extensions/   本地 Python 扩展加载与 hook 分发
├── llm/          provider、模型注册、消息转换、streaming
│   └── providers/  openai / anthropic / gemini 等具体实现
├── mcp/          MCP 配置、连接管理、工具适配与项目信任
├── sandbox/      local / workspace / Docker 执行环境与策略
├── skills/       skill 加载与读取
├── storage/      文件与 SQLite 应用状态存储
├── tools/        工具 base、registry 与 file/bash/web/memory/subagent/skill 实现
└── tui/          基于 prompt_toolkit 的全屏 TUI
tests/            行为与回归测试
```

依赖方向：`cli` / `tui` → `agent` → `tools` / `llm`；`llm` 不感知 CLI、会话文件和 UI；事件是层与层之间的边界。

---

## 🧪 开发与测试

```bash
# 运行全部测试
uv run pytest

# 运行单个文件
uv run pytest tests/spice/agent/test_loop.py

# 调试模式下运行 CLI
uv run spice --debug
```

测试约定：

- 优先覆盖**事件顺序、会话格式、provider 转换、工具执行、compaction、stdout 清洁度**等行为。
- 不冻结实现细节，避免大段 mock。
- TUI / CLI 共享逻辑应同时被两端的测试覆盖。

---

## 🔗 友链

[LINUX DO - 新的理想型社区](https://linux.do/)

---

## 📄 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
