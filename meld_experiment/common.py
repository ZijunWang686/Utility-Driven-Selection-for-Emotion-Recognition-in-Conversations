"""Shared text-only MELD data and prompt helpers; no ML dependencies."""
import json
from pathlib import Path

LABELS = ('neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger')
SYSTEM = 'Recognize the emotion of the target utterance in its conversational context. Return exactly one label from: ' + ', '.join(LABELS) + '. Do not add explanations.'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def messages(row, history=None):
    history = row['history'] if history is None else history
    context = '\n'.join(f"{r['speaker']}: {r['text']}" for r in history) or '(No earlier utterances.)'
    target = row['current']
    user = ("You are an expert in Emotion Recognition in Conversations.\n\n"
            "Identify the emotion of the target speaker in the current utterance based on the preceding conversation.\n\n"
            "Choose exactly one label:\nneutral, joy, surprise, anger, sadness, disgust, fear.\n\n"
            f"Conversation History:\n{context}\n\nTarget Speaker: {target['speaker']}\n"
            f"Current Utterance: {target['text']}\n\nOutput only the emotion label in lowercase.")
    return [{'role': 'user', 'content': user}]


def encode_prompt(tokenizer, row, budget, enable_thinking=None):
    """Drop oldest history first; never truncate the target or classification instruction."""
    history = list(row['history'])
    while True:
        options = {'tokenize': True, 'add_generation_prompt': True}
        if enable_thinking is not None:
            options['enable_thinking'] = enable_thinking
        tokens = tokenizer.apply_chat_template(messages(row, history), **options)
        if len(tokens) <= budget:
            return tokens, len(row['history']) - len(history)
        if not history:
            raise ValueError(f"{row['id']}: target alone exceeds prompt budget {budget}; increase max_length")
        history.pop(0)


def parse_prediction(text):
    # Strict parsing: never search an explanation for an arbitrary emotion word.
    label = text.strip().lower()
    return label if label in LABELS else '__invalid__'


def encode_supervised(tokenizer, row, max_length, answer_budget):
    target = tokenizer(row['label'] + tokenizer.eos_token, add_special_tokens=False)['input_ids']
    if len(target) > answer_budget:
        raise ValueError('Increase max_new_tokens to fit the label and EOS')
    prompt, _ = encode_prompt(tokenizer, row, max_length - answer_budget, enable_thinking=False)
    return {'input_ids': prompt + target, 'labels': [-100] * len(prompt) + target}


def keyed(rows):
    out = {}
    for row in rows:
        if not isinstance(row.get('id'), str) or not row['id'].strip() or row['id'] != row['id'].strip():
            raise ValueError('Sample id must be a nonempty string without surrounding whitespace')
        if row['id'] in out:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        out[row['id']] = row
    return out
