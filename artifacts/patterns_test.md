# Patterns — test

## braces
- regex: `\{[^}]*\}`
- match_count: 0

_examples: none_

## brackets
- regex: `\[[^\]]*\]`
- match_count: 0

_examples: none_

## parens
- regex: `\([^)]*\)`
- match_count: 2

### examples
- `0`: `ar-ma ú wa-bar-ra-tim qí-bi„-ma mup-pu-um aa a-lim(ki) i-li-kam`
- `1`: `i-na mup-pì-im aa a-lim(ki) ia-tù u„-mì-im a-nim ma-ma-an KÙ.AN i-aa-ú-mu-ni `

## angle
- regex: `<[^>]*>`
- match_count: 0

_examples: none_

## pipe
- regex: `\|[^|]*\|`
- match_count: 0

_examples: none_

## at_line
- regex: `(^|\n)\s*@\S+`
- match_count: 0

_examples: none_

## dollar_line
- regex: `(^|\n)\s*\$\s*\S+`
- match_count: 0

_examples: none_

## hash_line
- regex: `(^|\n)\s*#\S+`
- match_count: 0

_examples: none_

## percent_code
- regex: `%[a-zA-Z]{1,4}`
- match_count: 0

_examples: none_

## subscript_digits
- regex: `[₀₁₂₃₄₅₆₇₈₉]`
- match_count: 0

_examples: none_
