import ast


def parse_token_field(value):
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            return [int(x) for x in parsed]
        except (ValueError, SyntaxError, TypeError):
            return []
    if isinstance(value, (list, tuple, set)):
        return [int(x) for x in value]
    return []


def build_token_map(df, token_col):
    token_ids = set()
    for tokens in df[token_col].dropna().apply(parse_token_field):
        token_ids.update(tokens)
    return {token_id: i for i, token_id in enumerate(sorted(token_ids))}
