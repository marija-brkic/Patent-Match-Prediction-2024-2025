def extract_text(content_dict, text_type="full"):
    if text_type == "TA" or text_type == "title_abstract":
        title = content_dict.get("title", "")
        abstract = content_dict.get("pa01", "")
        return f"{title} {abstract}".strip()

    elif text_type == "claims":
        return " ".join([v for k, v in content_dict.items() if k.startswith('c-')])

    elif text_type == "tac1":
        title = content_dict.get("title", "")
        abstract = content_dict.get("pa01", "")
        first_claim = next((v for k, v in content_dict.items() if k.startswith('c-')), "")
        return f"{title} {abstract} {first_claim}".strip()

    elif text_type == "description":
        return " ".join([v for k, v in content_dict.items() if k.startswith('p')])

    elif text_type == "features":
        return content_dict.get("features", "")

    elif text_type == "full":
        fields = [content_dict.get("title", ""), content_dict.get("pa01", "")]
        fields += [v for k, v in content_dict.items() if k not in ("title", "pa01")]
        return " ".join(fields)

    return ""
