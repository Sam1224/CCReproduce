"""
Prompt templates for EVADE evaluation.
Implements both Single-Violation and All-in-One prompt designs from the paper.

Paper design:
- Single-Violation: short prompt, one specific rule, fine-grained reasoning
- All-in-One: long prompt, all overlapping rules merged, long-context reasoning
"""

POLICY_RULES = {
    "body_shaping": [
        "不得声称产品能够保证减重或改变体型",
        "不得在没有科学证据的情况下使用前后对比图片",
        "不得声称在特定短时间内产生效果",
    ],
    "height_growth": [
        "不得声称产品能在骨骺闭合后帮助成年人增高",
        "不得使用有关身高增长的误导性统计数据",
        "不得针对成年人宣传增高功效",
    ],
    "health_supplements": [
        "不得进行未经验证的医疗或治疗性声明",
        "不得声称能治愈、治疗或预防疾病",
        "不得引用临床研究而不提供适当引用",
    ],
    "medical_devices": [
        "不得在没有临床证据的情况下夸大设备功效",
        "不得声称获得尚未取得的监管批准",
        "不得使用暗示医疗有效性的用户证言",
    ],
    "weight_loss": [
        "不得在没有证据的情况下承诺具体的减重量",
        "不得声称快速减重效果（如一周减10kg）",
        "不得使用误导性的'天然'或'保证'声明",
    ],
    "skin_care": [
        "不得在没有批准的情况下声称具有药物级别的功效",
        "不得使用经过数字修改的前后图片",
        "不得声称能永久改变皮肤结构",
    ],
}

SYSTEM_PROMPT = (
    "你是一名专业的电商内容合规审核员，负责检测商品文案中的违规内容。"
    "请仔细分析以下商品文案，判断是否存在违规内容。"
    "违规内容包括：隐晦的夸大功效声明、误导性统计数据、暗示性疗效描述等规避性内容。"
)

SYSTEM_PROMPT_EN = (
    "You are a professional e-commerce content compliance reviewer. "
    "Your task is to detect evasive violation content in product descriptions. "
    "Evasive content includes: subtle exaggerated efficacy claims, misleading statistics, "
    "implied therapeutic effects, and other content that superficially complies with policies "
    "while covertly conveying prohibited claims."
)


def build_single_violation_prompt(
    text: str,
    category: str,
    rule: str | None = None,
    language: str = "zh",
) -> str:
    """
    Single-Violation task prompt.
    One specific rule tested against one sample.
    Paper: shorter context, fine-grained reasoning per rule.
    """
    if rule is None:
        rules = POLICY_RULES.get(category, ["不得进行未经验证的功效声明"])
        rule = rules[0]  # Use first rule if none specified

    if language == "zh":
        return (
            f"商品类目：{category.replace('_', ' ')}\n"
            f"平台规则：{rule}\n\n"
            f"商品文案：\n{text}\n\n"
            f"请判断上述商品文案是否违反了平台规则。"
            f"仅回答"违规"或"合规"，不需要额外解释。"
        )
    else:
        rule_en = rule  # Use as-is for English
        return (
            f"Product category: {category.replace('_', ' ')}\n"
            f"Policy rule: {rule_en}\n\n"
            f"Product description:\n{text}\n\n"
            f"Does this product description violate the policy rule? "
            f"Answer with only 'violation' or 'compliant'."
        )


def build_all_in_one_prompt(
    text: str,
    category: str,
    language: str = "zh",
) -> str:
    """
    All-in-One task prompt.
    All rules for the category merged into unified instructions.
    Paper: longer context, long-context reasoning, tests rule integration.
    """
    rules = POLICY_RULES.get(category, ["不得进行未经验证的功效声明"])
    rules_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules))

    if language == "zh":
        return (
            f"商品类目：{category.replace('_', ' ')}\n\n"
            f"平台合规政策（所有规则均须遵守）：\n{rules_text}\n\n"
            f"商品文案：\n{text}\n\n"
            f"请判断上述商品文案是否违反了上述任何一条平台政策规则。"
            f"注意：违规内容可能以隐晦、规避的方式出现，需要仔细分析。"
            f"仅回答"违规"或"合规"，不需要额外解释。"
        )
    else:
        rules_en_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules))
        return (
            f"Product category: {category.replace('_', ' ')}\n\n"
            f"Platform policies (ALL rules must be followed):\n{rules_en_text}\n\n"
            f"Product description:\n{text}\n\n"
            f"Does this description violate ANY of the platform policies listed above? "
            f"Note: Violations may appear in subtle or evasive forms. "
            f"Answer with only 'violation' or 'compliant'."
        )


def build_multimodal_prompt(
    text: str,
    category: str,
    task_type: str = "single_violation",
    language: str = "zh",
) -> tuple[str, str]:
    """
    Build prompt for multimodal (text+image) evaluation.
    Returns (system_prompt, user_prompt) tuple.
    The image should be passed separately to the VLM API.
    """
    sys = SYSTEM_PROMPT if language == "zh" else SYSTEM_PROMPT_EN

    if task_type == "single_violation":
        rules = POLICY_RULES.get(category, [])
        rule = rules[0] if rules else "不得进行未经验证的功效声明"
        user = build_single_violation_prompt(text, category, rule, language)
    else:
        user = build_all_in_one_prompt(text, category, language)

    # Add image instruction for multimodal
    if language == "zh":
        user = "请同时分析商品图片和文案内容，检测是否存在违规。\n\n" + user
    else:
        user = "Please analyze both the product image and text for violations.\n\n" + user

    return sys, user
