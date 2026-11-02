import csv
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import messages, parse_prediction, encode_prompt, encode_supervised
from prepare_text import prepare
from score import evaluate
from download_meld import safe_extract
from zero_shot_meld import final_answer, find_csv, label_constraint
from caption_meld import parse_visual_response


class PipelineTests(unittest.TestCase):
    def test_visual_response_requires_unambiguous_structure(self):
        self.assertEqual(parse_visual_response('{"target_status":"visible","description":"Hands raised."}'), ('visible', 'Hands raised.'))
        self.assertEqual(parse_visual_response('{"target_status":"uncertain","description":""}'), ('uncertain', ''))
        for text in ('{"target_status":"uncertain","description":"A man."}',
                     '{"target_status":"visible","description":""}',
                     '{"target_status":"maybe","description":""}'):
            with self.assertRaises(ValueError):
                parse_visual_response(text)

    def test_label_constraint_allows_only_complete_labels(self):
        from common import LABELS
        class Tokens(list):
            def __getitem__(self, key):
                value = super().__getitem__(key)
                return Tokens(value) if isinstance(key, slice) else value
            def tolist(self):
                return list(self)
        class Tokenizer:
            eos_token_id = 99
            def encode(self, label, add_special_tokens=False):
                return [1, LABELS.index(label) + 2]
        allowed = label_constraint(Tokenizer(), 2, 4)
        self.assertEqual(allowed(0, Tokens([8, 8])), [1])
        self.assertEqual(allowed(0, Tokens([8, 8, 1])), list(range(2, 9)))
        self.assertEqual(allowed(0, Tokens([8, 8, 1, 2])), [99])
        with self.assertRaises(ValueError):
            label_constraint(Tokenizer(), 2, 2)

    def fixture(self, directory):
        path = Path(directory) / 'data.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Utterance', 'Emotion'])
            writer.writeheader()
            for did, uid in [(0, i) for i in reversed(range(15))] + [(1, 0)]:
                writer.writerow({'Dialogue_ID': did, 'Utterance_ID': uid, 'Speaker': 'Speaker',
                                 'Utterance': f'text_{did}_{uid}', 'Emotion': 'neutral'})
        return path

    def test_history_is_causal_bounded_and_dialogue_local(self):
        with tempfile.TemporaryDirectory() as d:
            inputs, gold, train = prepare(self.fixture(d), 'train')
        self.assertEqual(len(inputs), 16)
        self.assertEqual([h['utterance_id'] for h in inputs[14]['history']], list(range(2, 14)))
        self.assertEqual(inputs[-1]['history'], [])
        self.assertNotIn('label', inputs[0])
        self.assertEqual(train[0]['label'], gold[0]['label'])
        prompt = messages(inputs[5])[0]['content']
        self.assertIn('text_0_5', prompt)
        self.assertNotIn('text_0_6', prompt)

    def test_split_is_part_of_id_and_zero_window(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.fixture(d)
            train, _, _ = prepare(path, 'train', 0)
            dev, _, _ = prepare(path, 'dev', 0)
        self.assertNotEqual(train[0]['id'], dev[0]['id'])
        self.assertTrue(all(not r['history'] for r in train))

    def test_truncation_keeps_current(self):
        class Tokenizer:
            def apply_chat_template(self, msgs, **kwargs):
                return list(msgs[0]['content'])
        row = {'id': 'x', 'current': {'speaker': 'A', 'text': 'TARGET'},
               'history': [{'speaker': 'B', 'text': 'old' * 100}]}
        tokens, dropped = encode_prompt(Tokenizer(), row, 400)
        self.assertEqual(dropped, 1)
        self.assertIn('TARGET', ''.join(tokens))
        with self.assertRaises(ValueError):
            encode_prompt(Tokenizer(), row, 1)

    def test_invalid_outputs_not_silently_parsed(self):
        self.assertEqual(parse_prediction(' ANGER\n'), 'anger')
        self.assertEqual(parse_prediction('not anger, maybe joy'), '__invalid__')

    def test_thinking_parser_uses_final_answer_only(self):
        self.assertEqual(final_answer('<think>perhaps anger</think> joy<|im_end|>', 'think'), ('joy', 'joy'))
        self.assertEqual(final_answer('<think>anger', 'think')[1], '__invalid__')
        self.assertEqual(final_answer('joy because context', 'no_think')[1], '__invalid__')

    def test_data_root_resolution_and_thinking_switch(self):
        class Tokenizer:
            def apply_chat_template(self, msgs, **kwargs):
                self.kwargs = kwargs
                return [1, 2, 3]
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'MELD.Raw'
            raw.mkdir()
            csv_path = raw / 'dev_sent_emo.csv'
            csv_path.touch()
            self.assertEqual(find_csv(d, 'dev'), csv_path.resolve())
        tokenizer = Tokenizer()
        row = {'id': 'dev/a', 'current': {'speaker': 'A', 'text': 'Hello'}, 'history': []}
        encode_prompt(tokenizer, row, 10, enable_thinking=False)
        self.assertFalse(tokenizer.kwargs['enable_thinking'])

    def test_training_only_supervises_answer_and_eos(self):
        class Tokenizer:
            eos_token = '<end>'
            def __call__(self, text, **kwargs):
                self.last_target = text
                return {'input_ids': [10, 99]}
            def apply_chat_template(self, msgs, **kwargs):
                return [1, 2, 3, 4]
        tokenizer = Tokenizer()
        row = {'id': 'train/a', 'label': 'anger', 'current': {'speaker': 'A', 'text': 'target'}, 'history': []}
        result = encode_supervised(tokenizer, row, 32, 8)
        self.assertEqual(result['labels'], [-100, -100, -100, -100, 10, 99])
        self.assertEqual(result['input_ids'], [1, 2, 3, 4, 10, 99])
        self.assertEqual(tokenizer.last_target, 'anger<end>')

    def test_metrics_align_by_id_and_penalize_invalid(self):
        gold = [{'id': 'a', 'label': 'anger'}, {'id': 'b', 'label': 'joy'}]
        predicted = [{'id': 'b', 'prediction': '__invalid__'}, {'id': 'a', 'prediction': 'anger'}]
        score = evaluate(gold, predicted)
        self.assertEqual(score['accuracy'], 50)
        self.assertEqual(score['weighted_f1'], 50)
        self.assertEqual(score['invalid_predictions'], 1)
        self.assertEqual(sum(map(sum, score['confusion_matrix'])), 2)

    def test_missing_or_duplicate_predictions_fail(self):
        gold = [{'id': 'a', 'label': 'anger'}, {'id': 'b', 'label': 'joy'}]
        pred = [{'id': 'a', 'prediction': 'anger'}]
        with self.assertRaises(ValueError):
            evaluate(gold, pred)
        with self.assertRaises(ValueError):
            evaluate(gold, pred + pred)

    def test_extraction_preserves_regular_file(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / 'test.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                member = tarfile.TarInfo('train/clip.mp4')
                member.size = 4
                tar.addfile(member, io.BytesIO(b'data'))
            safe_extract(archive, Path(d) / 'out')
            self.assertEqual((Path(d) / 'out/train/clip.mp4').read_bytes(), b'data')

    def test_extraction_rejects_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            for name, symlink in [('../escape', False), ('link', True)]:
                archive = Path(d) / 'bad.tar.gz'
                with tarfile.open(archive, 'w:gz') as tar:
                    member = tarfile.TarInfo(name)
                    if symlink:
                        member.type = tarfile.SYMTYPE
                        member.linkname = '../escape'
                    tar.addfile(member)
                with self.assertRaises(ValueError):
                    safe_extract(archive, Path(d) / 'out')
            self.assertFalse((Path(d) / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
