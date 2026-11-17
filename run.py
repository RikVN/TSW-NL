#!/usr/bin/env python3

"""Run TSW-NL zero-shot with a Hugging Face chat model, in clue order 123."""

import argparse
import json
from pathlib import Path

from evaluate import ROOT, load_jsonl, parse_model_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Hugging Face model ID or local model directory')
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.jsonl'))
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--limit', type=int, help='Run only the first N test items for a smoke check')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        parser.error('--max-new-tokens and --limit must be positive')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.chat_template:
        parser.error('This runner requires a model with a chat template.')
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype='auto', device_map='auto')
    model.eval()
    system = (ROOT / 'prompts/system.txt').read_text(encoding='utf-8').strip()
    template = (ROOT / 'prompts/user.txt').read_text(encoding='utf-8').strip()
    rows = load_jsonl(ROOT / 'data/test.jsonl')
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        for index, row in enumerate(rows, 1):
            clues = [row[f'clue{i}'].strip() for i in (1, 2, 3)]
            clues = [clue[0].lower() + clue[1:] for clue in clues]
            puzzle = f'Het is {clues[0]}, {clues[1]}, en {clues[2]}'
            messages = [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': template.replace('{puzzle}', puzzle)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            answer, warning = parse_model_answer(raw, 'json')
            result = {
                'id': row['id'], 'answer': answer, 'raw_response': raw,
                'parse_warning': warning, 'model': args.model,
                'model_revision': getattr(model.config, '_commit_hash', None),
                'settings': {'shot': 'zero', 'clue_order': '123', 'do_sample': False,
                             'max_new_tokens': args.max_new_tokens, 'seed': 42,
                             'dtype': str(model.dtype)},
                'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
                'messages': messages,
            }
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            output.flush()
            print(f'{index}/{len(rows)}: {row["id"]}', flush=True)
    print(f'Saved predictions to {args.output}')


if __name__ == '__main__':
    main()
