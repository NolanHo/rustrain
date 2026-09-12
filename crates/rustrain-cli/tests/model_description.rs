//! D1 的行为门禁：描述文件能不能表达一个模型、并展开成全局 Plan。
//!
//! 全部走行为层 —— 只调用 CLI 可执行文件，只看退出码 / stdout / stderr，不碰任何 Rust 内部 API。
//! 契约来源：`docs/design/qwen36-text/spec.md` C1（行为与稳定接口）+ D1（验收与证据），
//! 语法与语义来源：`docs/design/model-description.md` §3（四个部分）+ §4（`expand` 语义）。
//!
//! **这是 TDD 的 Red 步骤**：`plan explain` 目前还没有 `--model` 参数，这六个用例此刻必须失败。
//!
//! fixture 里三处契约没写死、由本文件做最小解释的地方（详见交付报告"契约里我没看懂的地方"）：
//!
//! 1. 模型目录里的描述文件名取 `model.json`：契约只说 `--model <dir>` 是模型目录、目录含
//!    `config.json`，没有规定描述文件叫什么、放在哪。
//! 2. 描述顶层的 `inputs` 段按模板 `inputs`/`slots` 的同一套子结构书写
//!    （`{ "shape": [...], "kind": "..." }`）：§3.3 说"首项的 inputs 来自描述的 `inputs` 段"，
//!    但 §3 又只列了四个顶层键，`inputs` 段的语法没有定义。
//! 3. fixture 的 `binding` 只写 `slot` / `source`，不写 `axes` / `transform`：C1 说展开后的全局
//!    Plan `layout` 全 `Replicate`，那么最小形态就是"每个 weight slot 有 binding、没有任何切分"。

use std::path::{Path, PathBuf};
use std::process::Command;

/// 被测可执行文件。
///
/// `CARGO_BIN_EXE_<target>` 由 cargo 注入集成测试，`<target>` 是**二进制 target 名**。本包当前的
/// target 叫 `rustrain-cli`（`Cargo.toml` 里没有 `[[bin]] name = "rustrain"`），而设计文档里的命令行
/// 一律写成 `rustrain ...`。两个名字都接受，这样把 target 改名成 `rustrain` 不会把门禁变成编译错误。
fn cli_binary() -> &'static str {
    option_env!("CARGO_BIN_EXE_rustrain")
        .or(option_env!("CARGO_BIN_EXE_rustrain-cli"))
        .expect("cargo 没有注入 CARGO_BIN_EXE_<bin>：检查 rustrain-cli 的二进制 target 名")
}

/// fixture 根目录 `crates/rustrain-cli/tests/fixtures/model-desc/<name>`。
///
/// 每个子目录都是一个完整的模型目录（`config.json` + `model.json`），可以直接交给 `--model`。
fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/model-desc")
        .join(name)
}

/// 一次 CLI 调用的可观测量。
///
/// stdout / stderr 保留原始字节：确定性契约要的是"逐字节相同"，转成 `String` 再比会丢掉这个性质。
struct Run {
    code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl Run {
    fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    fn stderr_text(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }

    /// 失败用例的断言信息里带上两份输出，否则只有一行 "assertion failed" 无从定位。
    fn dump(&self) -> String {
        format!(
            "exit = {:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            self.code,
            self.stdout_text(),
            self.stderr_text()
        )
    }

    fn expect_success(&self) {
        assert_eq!(self.code, Some(0), "期望退出码 0\n{}", self.dump());
    }

    fn expect_failure(&self) {
        assert!(
            self.code != Some(0),
            "期望非 0 退出码，实际 {:?}\n{}",
            self.code,
            self.dump()
        );
    }
}

/// 跑一次 `rustrain plan explain --model <model-dir> --json`。
fn plan_explain_json(model_dir: &Path) -> Run {
    let output = Command::new(cli_binary())
        .args(["plan", "explain", "--model"])
        .arg(model_dir)
        .arg("--json")
        .output()
        .unwrap_or_else(|e| panic!("启动 {} 失败: {e}", cli_binary()));
    Run {
        code: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    }
}

/// 契约只写了"含 `nodes` 与 `slots` 字段（非空）"，没说这两个字段是数组还是计数。
/// 两种形态都算满足：数组看长度，数字看是否大于 0。
fn assert_non_empty_field(doc: &serde_json::Value, key: &str) {
    let field = doc
        .get(key)
        .unwrap_or_else(|| panic!("plan JSON 缺少 `{key}` 字段: {doc}"));
    match field {
        serde_json::Value::Array(items) => assert!(!items.is_empty(), "`{key}` 是空数组"),
        serde_json::Value::Number(n) => {
            assert!(n.as_u64().is_some_and(|v| v > 0), "`{key}` = {n}，不是非空计数");
        }
        other => panic!("`{key}` 既不是数组也不是非空计数: {other}"),
    }
}

/// 契约 1：合成描述 + 配套 `config.json` 必须能展开，`--json` 给出含 `nodes` / `slots` 的 JSON。
#[test]
fn plan_explain_model_dir_emits_json_with_nodes_and_slots() {
    let run = plan_explain_json(&fixture("ok"));
    run.expect_success();

    let doc: serde_json::Value = serde_json::from_str(&run.stdout_text())
        .unwrap_or_else(|e| panic!("stdout 不是可解析 JSON: {e}\n{}", run.dump()));
    assert_non_empty_field(&doc, "nodes");
    assert_non_empty_field(&doc, "slots");
}

/// 契约 2：同一输入连跑两次，stdout 逐字节相同（§4.1 "同一份描述 + 同一份 config 必然得到同一个 Plan"）。
#[test]
fn plan_explain_is_byte_deterministic() {
    let model_dir = fixture("ok");
    let first = plan_explain_json(&model_dir);
    let second = plan_explain_json(&model_dir);
    first.expect_success();
    second.expect_success();

    // 空 stdout 逐字节相同是平凡真，不构成确定性证据：先要求真的产出了东西。
    assert!(!first.stdout.is_empty(), "stdout 为空\n{}", first.dump());
    assert_eq!(
        first.stdout, second.stdout,
        "两次运行的 stdout 必须逐字节相同\n第一次: {}\n第二次: {}",
        first.stdout_text(),
        second.stdout_text()
    );
}

/// 契约 3：两个 template 实例前缀相同（`twin`）→ 报错，并指出冲突的名字。
#[test]
fn duplicate_instance_prefix_is_rejected_naming_the_name() {
    let run = plan_explain_json(&fixture("duplicate-prefix"));
    run.expect_failure();

    // 两个 stack 项都用前缀 `twin`，于是 `twin.w` / `twin.y` 被声明两次；
    // 无论实现报的是 slot 名还是实例名前缀，都应带上这个唯一可辨的 token。
    assert!(
        run.stderr_text().contains("twin"),
        "stderr 必须指出冲突的名字（twin）\n{}",
        run.dump()
    );
}

/// 契约 4：`params` 里 `cyc_a = cyc_b + 1`、`cyc_b = cyc_a + 1` → 报错，并指出环。
#[test]
fn cyclic_params_are_rejected_naming_the_cycle() {
    let run = plan_explain_json(&fixture("cyclic-params"));
    run.expect_failure();

    // C1 要求"指出冲突双方"：环的两个端点都得出现在 stderr 里。
    assert!(
        run.stderr_text().contains("cyc_a"),
        "stderr 必须指出环的端点 cyc_a\n{}",
        run.dump()
    );
    assert!(
        run.stderr_text().contains("cyc_b"),
        "stderr 必须指出环的端点 cyc_b\n{}",
        run.dump()
    );
}

/// 契约 5：有一条 `binding` 的 `slot` 模式匹配不到任何 slot → 报错，并指出那个模式。
#[test]
fn binding_matching_no_slot_is_rejected_naming_the_pattern() {
    let run = plan_explain_json(&fixture("unmatched-binding"));
    run.expect_failure();

    assert!(
        run.stderr_text().contains("layers.*.no_such_weight"),
        "stderr 必须指出未命中的 slot 模式 `layers.*.no_such_weight`\n{}",
        run.dump()
    );
}

/// 契约 6：`--model` 指向不存在的目录 → 非 0 退出，信息可读（不是 panic backtrace）。
#[test]
fn missing_model_directory_fails_with_a_readable_error() {
    let missing = fixture("no-such-model-dir");
    assert!(!missing.exists(), "这个 fixture 不该存在: {}", missing.display());

    let run = plan_explain_json(&missing);
    run.expect_failure();
    assert!(
        !run.stderr.is_empty(),
        "stderr 为空，用户看不到任何提示\n{}",
        run.dump()
    );
    assert!(
        !run.stderr_text().contains("panicked at"),
        "输入不存在必须是可读的报错，不是 panic\n{}",
        run.dump()
    );
    // 非 0 退出也可能只是参数解析失败（clap 不认识 `--model`）：那种失败不证明命令诊断了缺失的输入。
    assert!(
        !run.stderr_text().contains("unexpected argument"),
        "错误来自参数解析而不是缺失的模型目录\n{}",
        run.dump()
    );
    assert!(
        run.stderr_text().contains("no-such-model-dir"),
        "报错信息应点名缺失的目录\n{}",
        run.dump()
    );
}
