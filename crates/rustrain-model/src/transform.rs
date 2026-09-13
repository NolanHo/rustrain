//! The `transform` vocabulary of a `binding` (C6, `docs/design/model-description.md` §3.4).
//!
//! Exactly two verbs: `transpose(i, j)` and `slice(dim, start, len)`. Splitting a fused storage is
//! the binding's **`split` field**, not a transform — the earlier `take` / `concat` /
//! `split(dim, sizes)` spellings are gone, and an unknown verb is a description error rather than
//! something a later stage may `skip`.
//!
//! One parser, two users: [`crate::expand`] validates a description with it (so a wrong verb is
//! caught while expanding, with the binding named), and the loading check evaluates the parsed
//! steps against a checkpoint shape. A second parser on the loader side would be a second source
//! for the same fact.

/// One parsed `transform` step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// `transpose(i, j)`: swap two axes. Negative axes count from the end.
    Transpose { i: i64, j: i64 },
    /// `slice(dim, start, len)`: keep `len` positions from `start` along `dim`. Negative `dim`
    /// counts from the end.
    Slice { dim: i64, start: i64, len: i64 },
}

/// The whole vocabulary, in the spelling a description writes (C6). Nothing else is accepted.
pub const TRANSFORM_VERBS: [&str; 2] = ["transpose", "slice"];

/// Parse one `transform` step: `<verb>(<integer>[, <integer>[, <integer>]])`.
///
/// The error names the step, the offending verb (or argument) and the vocabulary; the caller adds
/// the binding it came from.
pub fn parse_transform(text: &str) -> Result<Transform, String> {
    let (verb, rest) = text
        .split_once('(')
        .ok_or_else(|| format!("transform `{text}` is not `<verb>(<args>)`"))?;
    // The verb first: `take(model.up.weight)` has to be reported as the verb it is, not as an
    // argument list that happens to hold a tensor name.
    if !TRANSFORM_VERBS.contains(&verb) {
        return Err(format!(
            "transform `{text}`: unknown verb `{verb}`; the vocabulary is {} and nothing else \
             (C6; splitting a fused storage is the binding's `split` field, not a transform)",
            TRANSFORM_VERBS.join(", ")
        ));
    }
    let args = rest
        .strip_suffix(')')
        .ok_or_else(|| format!("transform `{text}` is missing its closing `)`"))?;
    let numbers = args
        .split(',')
        .map(|arg| arg.trim().parse::<i64>())
        .collect::<Result<Vec<i64>, _>>()
        .map_err(|_| {
            format!(
                "transform `{text}`: every argument must be an integer, got `{}`",
                args.trim()
            )
        })?;

    match verb {
        "transpose" => match numbers[..] {
            [i, j] => Ok(Transform::Transpose { i, j }),
            _ => Err(format!(
                "transform `{text}`: `transpose` takes exactly two dimensions, got {}",
                numbers.len()
            )),
        },
        _ => match numbers[..] {
            [dim, start, len] if start >= 0 && len > 0 => Ok(Transform::Slice { dim, start, len }),
            [_, start, len] => Err(format!(
                "transform `{text}`: `slice` needs `start >= 0` and `len > 0`, got start = {start}, \
                 len = {len}"
            )),
            _ => Err(format!(
                "transform `{text}`: `slice` takes exactly three arguments (`dim, start, len`), got \
                 {}",
                numbers.len()
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_verbs_parse() {
        assert_eq!(
            parse_transform("transpose(0,1)"),
            Ok(Transform::Transpose { i: 0, j: 1 })
        );
        assert_eq!(
            parse_transform("transpose(-1, 2)"),
            Ok(Transform::Transpose { i: -1, j: 2 })
        );
        assert_eq!(
            parse_transform("slice(1, 0, 96)"),
            Ok(Transform::Slice {
                dim: 1,
                start: 0,
                len: 96
            })
        );
    }

    /// C6 took these three verbs out; a description that still writes one is wrong, and wrong
    /// loudly — never a `skip` at load-check time.
    #[test]
    fn a_removed_or_unknown_verb_is_an_error_naming_it() {
        for step in ["take(model.up.weight)", "concat(0)", "split(1, [96, 64])", "flip(0)"] {
            let error = parse_transform(step).unwrap_err();
            let verb = step.split_once('(').unwrap().0;
            assert!(error.contains(verb), "{error}");
            assert!(error.contains(step), "{error}");
        }
        let error = parse_transform("take(model.up.weight)").unwrap_err();
        assert!(error.contains("transpose, slice"), "{error}");
    }

    #[test]
    fn a_malformed_argument_list_is_an_error_not_a_default() {
        assert!(parse_transform("transpose").unwrap_err().contains("`<verb>(<args>)`"));
        assert!(parse_transform("transpose(0,1").unwrap_err().contains("closing"));
        assert!(parse_transform("transpose(0)").unwrap_err().contains("exactly two"));
        assert!(parse_transform("transpose(a,b)").unwrap_err().contains("integer"));
        assert!(parse_transform("slice(1,0)").unwrap_err().contains("exactly three"));
        assert!(parse_transform("slice(1,-1,4)").unwrap_err().contains("start >= 0"));
        assert!(parse_transform("slice(1,0,0)").unwrap_err().contains("len > 0"));
    }
}
