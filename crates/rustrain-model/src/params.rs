//! `params`：值的唯一来源（§3.1），按依赖序求值，环 → 报错（§4.1 第 1 步）。
//!
//! 三种形式：`{"from": "text_config.hidden_size"}` 从 `config.json` 取（可带 `default`）、
//! `{"expr": "2 * heads * head_dim"}` 参数表达式、以及列表字面量（逐层类型）。
//! **不做"标量 → 结构"的派生**：需要派生时由生成器算成显式列表写进描述。

use std::collections::{BTreeMap, BTreeSet};

use crate::ModelError;
use crate::desc::ParamSpec;

/// 一个参数值。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(i64),
    List(Vec<String>),
}

/// 求值完成的参数表。
#[derive(Debug, Clone)]
pub struct Params {
    values: BTreeMap<String, Value>,
}

impl Params {
    /// 按依赖序求值 `decls`，`from` 从 `config` 取值。
    pub fn resolve(
        decls: &BTreeMap<String, ParamSpec>,
        config: &serde_json::Value,
    ) -> Result<Self, ModelError> {
        let mut values: BTreeMap<String, Value> = BTreeMap::new();
        let mut pending: Vec<(String, String, Ast)> = Vec::new();

        for (name, spec) in decls {
            match spec {
                ParamSpec::List(items) => {
                    values.insert(name.clone(), Value::List(items.clone()));
                }
                ParamSpec::From(from) => {
                    let value = from_config(config, &from.from)
                        .or_else(|| from.default.map(Value::Int))
                        .ok_or_else(|| {
                            ModelError::Invalid(format!(
                                "param `{name}`: `{}` is missing from {} (or is neither an integer \
                                 nor a list of strings) and no `default` is given",
                                from.from,
                                crate::desc::CONFIG_FILE
                            ))
                        })?;
                    values.insert(name.clone(), value);
                }
                ParamSpec::Expr(expr) => {
                    let ast = parse_expr(&expr.expr)
                        .map_err(|e| ModelError::Invalid(format!("param `{name}`: {e}")))?;
                    pending.push((name.clone(), expr.expr.clone(), ast));
                }
            }
        }

        let mut remaining = pending;
        while !remaining.is_empty() {
            let mut next: Vec<(String, String, Ast)> = Vec::new();
            let mut resolved_any = false;
            for (name, text, ast) in remaining {
                let mut ready = true;
                for dep in ast.idents() {
                    match values.get(&dep) {
                        Some(Value::Int(_)) => {}
                        Some(Value::List(_)) => {
                            return Err(ModelError::Invalid(format!(
                                "param `{name}`: `{text}` uses `{dep}` as a number, but `{dep}` \
                                 is a list"
                            )));
                        }
                        None if decls.contains_key(&dep) => ready = false,
                        None => {
                            return Err(ModelError::Invalid(format!(
                                "param `{name}`: `{text}` references `{dep}`, which is not declared"
                            )));
                        }
                    }
                }
                if ready {
                    let value = ast.eval(&values).map_err(|e| {
                        ModelError::Invalid(format!("param `{name}`: `{text}`: {e}"))
                    })?;
                    values.insert(name, Value::Int(value));
                    resolved_any = true;
                } else {
                    next.push((name, text, ast));
                }
            }
            if next.is_empty() {
                break;
            }
            if !resolved_any {
                // 没有任何一个能求值 —— 剩下的必然成环。
                let cycle = find_cycle(&next);
                return Err(ModelError::Invalid(format!(
                    "params form a cycle: {}",
                    cycle.join(" -> ")
                )));
            }
            remaining = next;
        }

        Ok(Self { values })
    }

    /// 取一个整数参数。
    pub fn int(&self, name: &str) -> Result<i64, ModelError> {
        match self.values.get(name) {
            Some(Value::Int(v)) => Ok(*v),
            Some(Value::List(_)) => Err(ModelError::Invalid(format!(
                "`{name}` is a list, but an integer is required here"
            ))),
            None => Err(ModelError::Invalid(format!(
                "`{name}` is not a declared param"
            ))),
        }
    }

    /// 取一个列表参数。
    pub fn list(&self, name: &str) -> Result<&[String], ModelError> {
        match self.values.get(name) {
            Some(Value::List(items)) => Ok(items),
            Some(Value::Int(_)) => Err(ModelError::Invalid(format!(
                "`{name}` is an integer, but a list is required here"
            ))),
            None => Err(ModelError::Invalid(format!(
                "`{name}` is not a declared param"
            ))),
        }
    }

    /// 参数是否已求值；用于区分"是参数"和"是字面量"。
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }
}

/// 在未求值的参数里找一条环，返回 `[a, b, a]` 形式的名字序列。
fn find_cycle(pending: &[(String, String, Ast)]) -> Vec<String> {
    let deps: BTreeMap<&str, BTreeSet<String>> = pending
        .iter()
        .map(|(name, _, ast)| (name.as_str(), ast.idents()))
        .collect();

    let mut path: Vec<String> = Vec::new();
    for (start, _, _) in pending {
        if dfs(start, &deps, &mut path, &mut BTreeSet::new()) {
            return path;
        }
    }
    // 走不到这里：`pending` 非空且无可求值项时一定存在环。退化成一个可读的列表。
    pending.iter().map(|(name, _, _)| name.clone()).collect()
}

fn dfs(
    node: &str,
    deps: &BTreeMap<&str, BTreeSet<String>>,
    path: &mut Vec<String>,
    seen: &mut BTreeSet<String>,
) -> bool {
    path.push(node.to_string());
    if let Some(next) = deps.get(node) {
        for dep in next {
            if !deps.contains_key(dep.as_str()) {
                continue;
            }
            if path.iter().any(|p| p == dep) {
                path.push(dep.clone());
                return true;
            }
            if seen.insert(dep.clone()) && dfs(dep, deps, path, seen) {
                return true;
            }
        }
    }
    path.pop();
    false
}

/// 按点分路径在 `config.json` 里取值，并归一成 [`Value`]。
///
/// `from` 既能取整数（`hidden_size`），也能取字符串列表（`layer_types` —— 40 项逐层类型）。
/// 后者必须能取，否则描述里就得复制一份层类型列表，那是第二个事实来源。
fn from_config(config: &serde_json::Value, path: &str) -> Option<Value> {
    let value = config_at(config, path)?;
    if let Some(int) = value.as_i64() {
        return Some(Value::Int(int));
    }
    let list = value.as_array()?;
    let items: Option<Vec<String>> = list
        .iter()
        .map(|item| item.as_str().map(str::to_string))
        .collect();
    Some(Value::List(items?))
}

/// 按点分路径在 `config.json` 里取值。
fn config_at<'a>(config: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let mut cur = config;
    for segment in path.split('.') {
        cur = cur.get(segment)?;
    }
    Some(cur)
}

/// 参数表达式：整数、参数名、`+ - * / ( )`。
#[derive(Debug, Clone, PartialEq, Eq)]
enum Ast {
    Num(i64),
    Param(String),
    Bin(BinOp, Box<Ast>, Box<Ast>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl Ast {
    fn idents(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.collect_idents(&mut out);
        out
    }

    fn collect_idents(&self, out: &mut BTreeSet<String>) {
        match self {
            Ast::Num(_) => {}
            Ast::Param(name) => {
                out.insert(name.clone());
            }
            Ast::Bin(_, lhs, rhs) => {
                lhs.collect_idents(out);
                rhs.collect_idents(out);
            }
        }
    }

    fn eval(&self, values: &BTreeMap<String, Value>) -> Result<i64, String> {
        match self {
            Ast::Num(v) => Ok(*v),
            Ast::Param(name) => match values.get(name) {
                Some(Value::Int(v)) => Ok(*v),
                _ => Err(format!("`{name}` has no integer value")),
            },
            Ast::Bin(op, lhs, rhs) => {
                let a = lhs.eval(values)?;
                let b = rhs.eval(values)?;
                match op {
                    BinOp::Add => a.checked_add(b),
                    BinOp::Sub => a.checked_sub(b),
                    BinOp::Mul => a.checked_mul(b),
                    BinOp::Div => {
                        if b == 0 {
                            return Err("division by zero".to_string());
                        }
                        a.checked_div(b)
                    }
                }
                .ok_or_else(|| format!("integer overflow evaluating `{a} {op:?} {b}`"))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Num(i64),
    Ident(String),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(text: &str) -> Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '0'..='9' => {
                let mut digits = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() {
                        digits.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let value: i64 = digits
                    .parse()
                    .map_err(|_| format!("`{digits}` does not fit in an i64"))?;
                out.push(Token::Num(value));
            }
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut name = String::new();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_alphanumeric() || d == '_' {
                        name.push(d);
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push(Token::Ident(name));
            }
            '+' | '-' | '*' | '/' => {
                out.push(Token::Op(c));
                chars.next();
            }
            '(' => {
                out.push(Token::LParen);
                chars.next();
            }
            ')' => {
                out.push(Token::RParen);
                chars.next();
            }
            other => return Err(format!("unexpected character `{other}`")),
        }
    }
    Ok(out)
}

fn parse_expr(text: &str) -> Result<Ast, String> {
    let tokens = tokenize(text)?;
    if tokens.is_empty() {
        return Err("empty expression".to_string());
    }
    let mut parser = Parser { tokens, pos: 0 };
    let ast = parser.expr()?;
    if parser.pos != parser.tokens.len() {
        return Err(format!("trailing input in `{text}`"));
    }
    Ok(ast)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn expr(&mut self) -> Result<Ast, String> {
        let mut lhs = self.term()?;
        while let Some(Token::Op(op @ ('+' | '-'))) = self.peek() {
            let op = match op {
                '+' => BinOp::Add,
                _ => BinOp::Sub,
            };
            self.pos += 1;
            let rhs = self.term()?;
            lhs = Ast::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn term(&mut self) -> Result<Ast, String> {
        let mut lhs = self.factor()?;
        while let Some(Token::Op(op @ ('*' | '/'))) = self.peek() {
            let op = match op {
                '*' => BinOp::Mul,
                _ => BinOp::Div,
            };
            self.pos += 1;
            let rhs = self.factor()?;
            lhs = Ast::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn factor(&mut self) -> Result<Ast, String> {
        match self.peek().cloned() {
            Some(Token::Num(v)) => {
                self.pos += 1;
                Ok(Ast::Num(v))
            }
            Some(Token::Ident(name)) => {
                self.pos += 1;
                Ok(Ast::Param(name))
            }
            Some(Token::LParen) => {
                self.pos += 1;
                let inner = self.expr()?;
                match self.peek() {
                    Some(Token::RParen) => {
                        self.pos += 1;
                        Ok(inner)
                    }
                    _ => Err("missing `)`".to_string()),
                }
            }
            Some(other) => Err(format!("unexpected token {other:?}")),
            None => Err("unexpected end of expression".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> serde_json::Value {
        serde_json::json!({"text_config": {"hidden_size": 16, "num_hidden_layers": 3}})
    }

    fn decls(text: &str) -> BTreeMap<String, ParamSpec> {
        serde_json::from_str(text).unwrap()
    }

    #[test]
    fn expressions_are_evaluated_in_dependency_order() {
        let params = Params::resolve(
            &decls(
                r#"{
                    "inter": {"expr": "2 * hidden"},
                    "hidden": {"from": "text_config.hidden_size"},
                    "layers": {"from": "text_config.num_hidden_layers", "default": 1}
                }"#,
            ),
            &config(),
        )
        .unwrap();
        assert_eq!(params.int("inter").unwrap(), 32);
        assert_eq!(params.int("layers").unwrap(), 3);
    }

    #[test]
    fn missing_config_key_without_default_is_rejected() {
        let err = Params::resolve(
            &decls(r#"{"hidden": {"from": "text_config.nope"}}"#),
            &config(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("text_config.nope"), "{err}");
    }

    #[test]
    fn a_cycle_names_both_endpoints() {
        let err = Params::resolve(
            &decls(r#"{"cyc_a": {"expr": "cyc_b + 1"}, "cyc_b": {"expr": "cyc_a + 1"}}"#),
            &config(),
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("cyc_a") && text.contains("cyc_b"), "{text}");
        assert!(text.contains("cycle"), "{text}");
    }

    #[test]
    fn undeclared_reference_is_rejected() {
        let err = Params::resolve(&decls(r#"{"a": {"expr": "b + 1"}}"#), &config()).unwrap_err();
        assert!(err.to_string().contains("`b`"), "{err}");
    }
}
