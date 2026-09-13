//! Name patterns: the one syntax shared by `binding.source`, `binding.slot` / `targets[].slot`
//! and `ignore` (§3.4, C6).
//!
//! `*` and `{*}` match **exactly one** dotted segment and are *captured*, in order. A source's
//! captures are what a target slot pattern's wildcards are filled with — that shared capture is
//! how a concrete checkpoint tensor is paired with the slot it feeds, instead of zipping source
//! instances against `ResolvedBinding::slots` by position (C5 forbids the zip: `slots` is grouped
//! by target).
//!
//! `**` matches **any number** of segments, including none, and captures nothing. It is the one
//! extension `ignore` needs, to drop a whole subtree by declaration (`"model.visual.**"`).

/// Any number of segments, capturing nothing.
const DOUBLE_STAR: &str = "**";
/// Exactly one segment, captured.
const SINGLE_STAR: &str = "*";
/// Exactly one segment, captured — the spelling `binding.source` uses inside a segment list.
const BRACED_STAR: &str = "{*}";

/// Match `pattern` against `name`, capturing the single-segment wildcards left to right.
///
/// `None` when the pattern does not match. A pattern with no wildcard is an ordinary name
/// comparison (`"norm.weight"` == `"norm.weight"`).
pub fn match_name(pattern: &str, name: &str) -> Option<Vec<String>> {
    let pattern: Vec<&str> = pattern.split('.').collect();
    let name: Vec<&str> = name.split('.').collect();
    let mut captures = Vec::new();
    match_segments(&pattern, &name, &mut captures).then_some(captures)
}

/// `true` when `pattern` matches `name`; the captures are dropped.
pub fn matches(pattern: &str, name: &str) -> bool {
    match_name(pattern, name).is_some()
}

/// Fill `pattern`'s single-segment wildcards with `captures`, in order (§3.4's shared capture).
///
/// `None` when the two counts differ — the caller has a source instance whose captures say nothing
/// about this target.
pub fn apply_captures(pattern: &str, captures: &[String]) -> Option<String> {
    let mut taken = captures.iter();
    let mut out: Vec<String> = Vec::new();
    for segment in pattern.split('.') {
        if is_single_star(segment) {
            out.push(taken.next()?.clone());
        } else {
            out.push(segment.to_string());
        }
    }
    taken.next().is_none().then(|| out.join("."))
}

fn is_single_star(segment: &str) -> bool {
    segment == SINGLE_STAR || segment == BRACED_STAR
}

/// Recursive match with backtracking: `**` is greedy in nothing, it tries the shortest prefix
/// first, so captures stay deterministic.
fn match_segments(pattern: &[&str], name: &[&str], captures: &mut Vec<String>) -> bool {
    let Some((head, rest)) = pattern.split_first() else {
        return name.is_empty();
    };
    if *head == DOUBLE_STAR {
        return (0..=name.len()).any(|taken| {
            let mark = captures.len();
            if match_segments(rest, &name[taken..], captures) {
                true
            } else {
                captures.truncate(mark);
                false
            }
        });
    }
    let Some((first, tail)) = name.split_first() else {
        return false;
    };
    if is_single_star(head) {
        captures.push((*first).to_string());
        if match_segments(rest, tail, captures) {
            true
        } else {
            captures.pop();
            false
        }
    } else {
        *head == *first && match_segments(rest, tail, captures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_star_matches_exactly_one_segment() {
        assert!(matches("layers.*.q", "layers.0.q"));
        assert!(!matches("layers.*.q", "layers.0.1.q"));
        assert!(!matches("layers.*.q", "layers.0.q_norm"));
        assert!(!matches("layers.*.q", "layers.q"));
        assert!(matches("norm_in.w", "norm_in.w"));
        // The two spellings of "one segment" are the same wildcard.
        assert!(matches("layers.{*}.q", "layers.0.q"));
    }

    #[test]
    fn a_double_star_spans_any_number_of_segments() {
        assert!(matches("model.visual.**", "model.visual.patch_embed.proj.weight"));
        assert!(matches("model.visual.**", "model.visual.weight"));
        // Zero segments: the subtree's own name is matched too.
        assert!(matches("model.visual.**", "model.visual"));
        // A prefix that does not match is still a prefix that does not match.
        assert!(!matches("model.visual.**", "model.language_model.norm.weight"));
        assert!(matches("a.**.b", "a.b"));
        assert!(matches("a.**.b", "a.x.y.b"));
        assert!(!matches("a.**.b", "a.x.y.c"));
    }

    #[test]
    fn wildcards_capture_in_order() {
        assert_eq!(
            match_name("layers.{*}.mlp.{*}.weight", "layers.7.mlp.1.weight"),
            Some(vec!["7".to_string(), "1".to_string()])
        );
        // `**` captures nothing: only the single-segment wildcards count.
        assert_eq!(
            match_name("model.**.layers.{*}", "model.language_model.layers.3"),
            Some(vec!["3".to_string()])
        );
        assert_eq!(match_name("norm.weight", "norm.weight"), Some(Vec::new()));
        assert_eq!(match_name("norm.weight", "norm.bias"), None);
    }

    #[test]
    fn captures_fill_a_slot_pattern() {
        let captures = vec!["12".to_string()];
        assert_eq!(
            apply_captures("layers.*.input_layernorm", &captures).as_deref(),
            Some("layers.12.input_layernorm")
        );
        // A count mismatch is not a pairing.
        assert_eq!(apply_captures("layers.*.q", &[]), None);
        assert_eq!(
            apply_captures("layers.q", &captures),
            None,
            "a pattern with no wildcard cannot absorb a capture"
        );
    }
}
