//! Differential check of the pattern matcher against the implementation it replaced (F1).
//!
//! `crates/rustrain-model/src/pattern.rs` was rewritten from recursive backtracking to a
//! `segments(pattern) × segments(name)` algorithm so a pathological `**` pattern cannot take
//! exponential time (the rewrite's own test covers that bound). The rewrite silently dropped one
//! rule the old matcher had in its base case — `return name.is_empty()` — so a pattern **without**
//! `**` matched a *prefix* of the name: `s.w` matched `s.w.extra`, and `ignore: ["s.extra"]`
//! swallowed the whole `s.extra.*` subtree.
//!
//! The old matcher is reproduced here verbatim from `git show 460178f:crates/rustrain-model/src/
//! pattern.rs` (module `oracle`, with its original body and comments intact) and used as an
//! independent oracle: over 400 000 random `(pattern, name)` pairs the two implementations must
//! agree on the match set *and* on the captures. The seed is fixed, so any disagreement is
//! reproducible; the generator's vocabulary is small and repetitive on purpose, because prefix
//! collisions between short names are exactly the class of bug this pins down.

use rustrain_model::{match_name, matches};

/// The matcher as it stood before the polynomial rewrite (`460178f`), verbatim. The single rule
/// under test is the `return name.is_empty()` base case: a pattern with no `**` consumes the whole
/// name.
mod oracle {
    /// Any number of segments, capturing nothing.
    const DOUBLE_STAR: &str = "**";
    /// Exactly one segment, captured.
    const SINGLE_STAR: &str = "*";
    /// Exactly one segment, captured — the spelling `binding.source` uses inside a segment list.
    const BRACED_STAR: &str = "{*}";

    /// `None` when the pattern does not match.
    pub fn match_name(pattern: &str, name: &str) -> Option<Vec<String>> {
        let pattern: Vec<&str> = pattern.split('.').collect();
        let name: Vec<&str> = name.split('.').collect();
        let mut captures = Vec::new();
        match_segments(&pattern, &name, &mut captures).then_some(captures)
    }

    pub fn matches(pattern: &str, name: &str) -> bool {
        match_name(pattern, name).is_some()
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
}

/// splitmix64: a fixed, self-contained generator, so the corpus is reproducible without adding a
/// dependency and without relying on the standard library's unspecified `RandomState`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn pick<'a>(&mut self, options: &[&'a str]) -> &'a str {
        options[self.below(options.len())]
    }
}

/// The pattern vocabulary: literals that share prefixes (`s` vs `sx`), plus every wildcard
/// spelling. `**` is capped by the generator below — the oracle is exponential in the number of
/// double stars, and this corpus is about the no-`**` rule, not about the bound the rewrite fixed.
const PATTERN_SEGMENTS: [&str; 10] = [
    "s", "w", "extra", "deep", "model", "visual", "sx", "*", "{*}", "**",
];

/// The name vocabulary: names built from the same literals, so `s.w` / `s.w.extra` / `s.wx`
/// near-misses are common rather than exotic.
const NAME_SEGMENTS: [&str; 8] = ["s", "w", "extra", "deep", "model", "visual", "layers", "0"];

fn random_pattern(rng: &mut Rng) -> String {
    loop {
        let length = 1 + rng.below(5);
        let segments: Vec<&str> = (0..length).map(|_| rng.pick(&PATTERN_SEGMENTS)).collect();
        if segments.iter().filter(|s| **s == "**").count() <= 2 {
            return segments.join(".");
        }
    }
}

fn random_name(rng: &mut Rng) -> String {
    let length = 1 + rng.below(6);
    (0..length)
        .map(|_| rng.pick(&NAME_SEGMENTS))
        .collect::<Vec<_>>()
        .join(".")
}

/// The pairs the regression is about, always in the corpus: a literal pattern against a longer,
/// equal-length and adjacent name.
const CORPUS: [(&str, &str); 16] = [
    ("s.w", "s.w"),
    ("s.w", "s.w.extra"),
    ("s.w", "s.w."),
    ("s.w", "s.wx"),
    ("s.w", "s.wx.y"),
    ("s.w", "s"),
    ("s.w", "w.s"),
    ("s.*", "s.w.extra"),
    ("s.*", "s.w"),
    ("*", "s.w"),
    ("{*}", "s"),
    ("s.**", "s"),
    ("s.**", "s.w.extra"),
    ("s.**.w", "s.w.extra"),
    ("model.visual.**", "model.visual.weight"),
    ("**", "anything.at.all"),
];

#[test]
fn the_matcher_agrees_with_the_pre_rewrite_one_on_random_patterns_and_names() {
    const PAIRS: usize = 400_000;

    let mut rng = Rng(0x5EED_1234_ABCD_EF01);
    let mut matched = 0usize;
    let mut matched_by_both = 0usize;
    let mut disagreements: Vec<String> = Vec::new();
    let mut capture_disagreements: Vec<String> = Vec::new();

    for (pattern, name) in CORPUS {
        if matches(pattern, name) != oracle::matches(pattern, name) {
            disagreements.push(format!("`{pattern}` vs `{name}` (fixed corpus)"));
        }
    }

    for index in 0..PAIRS {
        let (pattern, name) = if index % 4 == 0 {
            // Every fourth pair is drawn from the hand-written corpus: the near-misses the
            // generator could in principle miss are guaranteed to be covered.
            let (pattern, name) = CORPUS[rng.below(CORPUS.len())];
            (pattern.to_string(), name.to_string())
        } else {
            (random_pattern(&mut rng), random_name(&mut rng))
        };

        let old = oracle::match_name(&pattern, &name);
        let new = match_name(&pattern, &name);
        if old.is_some() {
            matched += 1;
        }
        if old.is_some() && new.is_some() {
            matched_by_both += 1;
        }
        if old.is_some() != new.is_some() {
            if disagreements.len() < 10 {
                disagreements.push(format!("`{pattern}` vs `{name}`: old {old:?} new {new:?}"));
            }
        } else if old != new && capture_disagreements.len() < 10 {
            capture_disagreements.push(format!("`{pattern}` vs `{name}`: old {old:?} new {new:?}"));
        }
    }

    assert!(
        disagreements.is_empty(),
        "the match sets differ: {} disagreement(s), first: {:?}",
        disagreements.len(),
        disagreements
    );
    assert!(
        capture_disagreements.is_empty(),
        "the match sets agree but the captures do not: {:?}",
        capture_disagreements
    );
    eprintln!(
        "differential: {PAIRS} random pairs + {} fixed pairs, {matched} hit(s) (old), {matched_by_both} \
         hit(s) by both, 0 disagreement",
        CORPUS.len()
    );
}
