//! `params`: the single source of values (§3.1), evaluated in dependency order; a cycle is an
//! error (§4.1 step 1).
//!
//! Three forms: `{"from": "text_config.hidden_size"}` reads `config.json` (optionally with a
//! `default`), `{"expr": "2 * heads * head_dim"}` is a parameter expression, and a bare list is a
//! literal (per-layer types). **No "scalar → structure" derivation**: when one is needed the
//! generator computes it and writes an explicit list into the description.

use std::collections::{BTreeMap, BTreeSet};

use crate::ModelError;
use crate::desc::ParamSpec;

/// One parameter value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Int(i64),
    List(Vec<String>),
}

/// The fully evaluated parameter table.
#[derive(Debug, Clone)]
pub struct Params {
    values: BTreeMap<String, Value>,
}

impl Params {
    /// Evaluate `decls` in dependency order; `from` reads values out of `config`.
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
                    let value = match from_config(config, &from.from).map_err(|why| {
                        ModelError::Invalid(format!(
                            "param `{name}`: {why}; a `default` covers only a missing key, never a \
                             value this build cannot read"
                        ))
                    })? {
                        Some(value) => value,
                        None => match from.default {
                            Some(default) => Value::Int(default),
                            None => {
                                return Err(ModelError::Invalid(format!(
                                    "param `{name}`: `{}` is missing from {} and no `default` is \
                                     given",
                                    from.from,
                                    crate::desc::CONFIG_FILE
                                )));
                            }
                        },
                    };
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
                // Nothing was evaluable, so what is left must form a cycle.
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

    /// Read an integer parameter.
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

    /// Read a list parameter.
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

    /// Whether a parameter is already evaluated; used to tell a parameter from a literal.
    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }
}

/// Find a cycle among the unevaluated parameters; returns the name sequence as `[a, b, a]`.
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
    // Unreachable: a non-empty `pending` with nothing evaluable must contain a cycle. Degrade to a
    // readable list anyway.
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

/// Read a dotted path out of `config.json` and normalise it into a [`Value`].
///
/// `from` has to read both integers (`hidden_size`) and lists of strings (`layer_types` — 40
/// per-layer types). The list form is not optional: without it the description would have to carry
/// a copy of the layer-type table, which is a second source for the same fact.
///
/// `Ok(None)` means the key is absent (or `null`, §3.7 #16); `Err` means it is present with a
/// non-null value this build cannot read. The two must stay apart — see the caller.
fn from_config(config: &serde_json::Value, path: &str) -> Result<Option<Value>, String> {
    let Some(value) = config_at(config, path)? else {
        return Ok(None);
    };
    if let Some(int) = value.as_i64() {
        return Ok(Some(Value::Int(int)));
    }
    let found = match value {
        serde_json::Value::Array(list) => {
            let items: Option<Vec<String>> = list
                .iter()
                .map(|item| item.as_str().map(str::to_string))
                .collect();
            match items {
                Some(items) => return Ok(Some(Value::List(items))),
                None => "a list whose entries are not all strings".to_string(),
            }
        }
        // Unreachable while `config_at` reports a `null` segment as a missing key (§3.7 #16); the
        // arm stays so the match remains exhaustive over `serde_json::Value`.
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(_) => "a boolean".to_string(),
        serde_json::Value::Number(number) => format!("the number {number}"),
        serde_json::Value::String(text) => format!("the string {text:?}"),
        serde_json::Value::Object(_) => "an object".to_string(),
    };
    Err(format!(
        "`{path}` is present in {} but holds {found}, which is neither an integer nor a list of \
         strings",
        crate::desc::CONFIG_FILE
    ))
}

/// Look up a dotted path in `config.json`.
///
/// `Ok(None)` when a segment is absent **or `null`**: JSON `null` is the usual way of writing "not
/// configured" (`"rope_scaling": null`), so it is a missing key, not a value of the wrong type
/// (§3.7 #16) — a `default` covers it. `Err` when a segment on the way exists and holds a non-null
/// value that is not an object: the path is then unreadable rather than missing, and a `default`
/// must not cover it either.
fn config_at<'a>(
    config: &'a serde_json::Value,
    path: &str,
) -> Result<Option<&'a serde_json::Value>, String> {
    let mut cur = config;
    let mut walked: Vec<&str> = Vec::new();
    for segment in path.split('.') {
        let Some(object) = cur.as_object() else {
            let parent = walked.join(".");
            let parent = if parent.is_empty() {
                crate::desc::CONFIG_FILE
            } else {
                parent.as_str()
            };
            return Err(format!(
                "`{parent}` is not an object, so `{path}` cannot be read"
            ));
        };
        let Some(next) = object.get(segment) else {
            return Ok(None);
        };
        if next.is_null() {
            return Ok(None);
        }
        cur = next;
        walked.push(segment);
    }
    Ok(Some(cur))
}

/// A parameter expression: integers, parameter names, `+ - * / ( )`.
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
    fn missing_config_key_with_a_default_uses_it() {
        let params = Params::resolve(
            &decls(r#"{"seq": {"from": "text_config.max_position_embeddings", "default": 8}}"#),
            &config(),
        )
        .unwrap();
        assert_eq!(params.int("seq").unwrap(), 8);
    }

    /// A key that exists with a type this build cannot read is not a missing key: `default` must
    /// not paper over what `config.json` actually says.
    #[test]
    fn a_config_value_of_an_unreadable_type_is_not_defaulted() {
        let config = serde_json::json!({"text_config": {"rope_theta": 10000000.0}});
        let err = Params::resolve(
            &decls(r#"{"theta": {"from": "text_config.rope_theta", "default": 10000}}"#),
            &config,
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("text_config.rope_theta"), "{text}");
        assert!(text.contains("10000000"), "{text}");
        assert!(text.contains("default"), "{text}");
    }

    /// The same distinction one level up: a path that walks through a non-object is unreadable,
    /// not missing.
    #[test]
    fn a_from_path_through_a_non_object_is_not_defaulted() {
        let config = serde_json::json!({"text_config": {"rope_scaling": 5}});
        let err = Params::resolve(
            &decls(r#"{"factor": {"from": "text_config.rope_scaling.factor", "default": 1}}"#),
            &config,
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("text_config.rope_scaling"), "{text}");
        assert!(text.contains("not an object"), "{text}");
    }

    /// §3.7 #16: JSON `null` is how HF writes "not configured" (`"rope_scaling": null`), so it
    /// counts as a missing key — both when the path walks through it and when the path ends on it.
    /// Only a *non-null* value of the wrong type is an error.
    #[test]
    fn a_null_config_value_is_treated_as_missing() {
        let config = serde_json::json!({"text_config": {"rope_scaling": null, "sliding": null}});
        let params = Params::resolve(
            &decls(
                r#"{
                    "factor": {"from": "text_config.rope_scaling.factor", "default": 1},
                    "window": {"from": "text_config.sliding", "default": 4}
                }"#,
            ),
            &config,
        )
        .unwrap();
        assert_eq!(params.int("factor").unwrap(), 1);
        assert_eq!(params.int("window").unwrap(), 4);
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
