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
//! extension `ignore` needs, to drop a whole subtree by declaration (`"model.visual.**"`). The
//! match runs in `O(segments(pattern) × segments(name))`: both sides are free-form input, so the
//! cost of a pathological pattern has to stay polynomial.

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

/// `true` when `pattern` uses the multi-segment wildcard. Only `ignore` may (C6).
pub fn has_double_star(pattern: &str) -> bool {
    pattern.split('.').any(|segment| segment == DOUBLE_STAR)
}

/// Match a segment pattern against a segment list, appending the single-segment captures.
///
/// `**` takes the shortest prefix that still lets the rest match — the same choice the recursive
/// matcher made, so captures stay deterministic — but the work is bounded by
/// `segments(pattern) × segments(name)`, not by the number of `**`. A pattern is free-form input
/// from a description and a name is free-form input from a checkpoint, so exponential
/// backtracking is a denial of service, not a match.
fn match_segments(pattern: &[&str], name: &[&str], captures: &mut Vec<String>) -> bool {
    let chunks = chunks_between_stars(pattern);
    if chunks.len() == 1 {
        // No `**`: the pattern is an ordinary name of the same length.
        return match_chunk(chunks[0], name, 0, captures);
    }

    // `S0 ** S1 ** … ** Sm`: `S0` is anchored at the start, `Sm` at the end, and every middle
    // chunk is floated as far forward as it goes. Earliest is optimal: the `**` after a chunk can
    // always absorb the segments a later position would have consumed, so a match at the earliest
    // position leaves the most room for everything that follows.
    if !match_chunk(chunks[0], name, 0, captures) {
        return false;
    }
    let mut position = chunks[0].len();

    let last = chunks.len() - 1;
    for chunk in &chunks[1..last] {
        let limit = name.len().checked_sub(chunk.len());
        let mut start = position;
        loop {
            let mark = captures.len();
            if match_chunk(chunk, name, start, captures) {
                position = start + chunk.len();
                break;
            }
            captures.truncate(mark);
            match limit {
                Some(limit) if start < limit => start += 1,
                _ => return false,
            }
        }
    }

    let Some(start) = name.len().checked_sub(chunks[last].len()) else {
        return false;
    };
    start >= position && match_chunk(chunks[last], name, start, captures)
}

/// The `**`-free runs between the double stars: one chunk more than there are double stars.
fn chunks_between_stars<'a>(pattern: &'a [&'a str]) -> Vec<&'a [&'a str]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    for (index, segment) in pattern.iter().enumerate() {
        if *segment == DOUBLE_STAR {
            chunks.push(&pattern[start..index]);
            start = index + 1;
        }
    }
    chunks.push(&pattern[start..]);
    chunks
}

/// Match one `**`-free chunk at `name[start..]`, appending the single-segment captures it takes.
/// Every chunk segment consumes exactly one name segment, so the chunk's length is fixed.
fn match_chunk(chunk: &[&str], name: &[&str], start: usize, captures: &mut Vec<String>) -> bool {
    let Some(end) = start.checked_add(chunk.len()) else {
        return false;
    };
    if end > name.len() {
        return false;
    }
    let mark = captures.len();
    for (offset, segment) in chunk.iter().enumerate() {
        let candidate = name[start + offset];
        if is_single_star(segment) {
            captures.push(candidate.to_string());
        } else if *segment != candidate {
            captures.truncate(mark);
            return false;
        }
    }
    true
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

    #[test]
    fn a_double_star_is_recognised_by_the_segment_it_occupies() {
        assert!(has_double_star("model.visual.**"));
        assert!(!has_double_star("model.visual.*"));
        assert!(!has_double_star("model.visual.w**"));
        assert!(!has_double_star("model.visual.weight"));
    }

    /// Both sides of a match are unbounded input: a pattern from a description, a tensor name from
    /// a checkpoint. Backtracking over `**` was exponential (eight double stars against a
    /// 36-segment name took ~10 s, and this 48-segment shape never returned), so the bound is part
    /// of the contract this matcher has to keep.
    #[test]
    fn a_pathological_double_star_pattern_returns_promptly() {
        let name = std::iter::once("m".to_string())
            .chain((0..47).map(|index| format!("s{index}")))
            .collect::<Vec<_>>()
            .join(".");
        let pattern = format!("m.{}.nomatch", ["**"; 10].join("."));
        assert_eq!(pattern.split('.').filter(|s| *s == "**").count(), 10);
        assert_eq!(name.split('.').count(), 48);

        let started = std::time::Instant::now();
        assert!(!matches(&pattern, &name));
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "matching {pattern} against {name} took {elapsed:?}; the bound is polynomial"
        );
    }

    /// The shortest-prefix rule is observable through captures: the first `**` takes as little as
    /// it can while the rest still matches.
    #[test]
    fn a_double_star_takes_the_shortest_prefix_that_still_matches() {
        assert_eq!(
            match_name("a.**.{*}.c", "a.b.d.c").as_deref(),
            Some(["d".to_string()].as_slice())
        );
        // The capture may only come from where the `**` stopped, never from inside it.
        assert_eq!(
            match_name("a.**.c.**.{*}", "a.c.x.y").as_deref(),
            Some(["y".to_string()].as_slice())
        );
        assert_eq!(match_name("**.{*}", "p.q"), Some(vec!["q".to_string()]));
        assert_eq!(match_name("{*}.{*}", "p.q"), Some(vec!["p".to_string(), "q".to_string()]));
    }
}
