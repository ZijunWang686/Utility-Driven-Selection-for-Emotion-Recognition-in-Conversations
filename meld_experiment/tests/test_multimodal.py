import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multimodal_data import (MODES,MEDIA_FILES,load_split,mode_for_sample,multimodal_messages,encode_training_row)
from train_multimodal import MixedDataset


class FakeTokenizer:
    eos_token='<eos>'
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking'] is False
        return list(range(len(messages[0]['content'])))
    def __call__(self,text,**kwargs): return {'input_ids':[10,11]}


class MultimodalTests(unittest.TestCase):
    def row(self):
        return {'id':'train/dia0_utt0','history':[{'speaker':'B','text':'past'}],
                'current':{'speaker':'A','text':'target'},'label':'joy',
                'audio_description':'audio evidence','video_description':'visual evidence'}
    def test_modes_include_only_selected_evidence_and_no_gold(self):
        for mode in MODES:
            content=multimodal_messages(self.row(),mode)[0]['content']
            self.assertEqual('Audio Description:' in content,mode in ('T+A','T+A+V'))
            self.assertEqual('Visual Description:' in content,mode in ('T+V','T+A+V'))
            self.assertNotIn("'label'",content)
            self.assertIn('target',content)
        self.assertIn('[unavailable]',multimodal_messages(dict(self.row(),video_description=None),'T+V')[0]['content'])
    def test_mask_and_epoch_sampling(self):
        row=self.row(); cfg={'seed':42,'max_length':2048,'max_new_tokens':16}
        encoded=encode_training_row(FakeTokenizer(),row,'T+A',2048,16)
        self.assertEqual(encoded['labels'][-2:],[10,11])
        self.assertTrue(all(i==-100 for i in encoded['labels'][:-2]))
        dataset=MixedDataset([row],FakeTokenizer(),cfg)
        self.assertEqual(dataset[0],dataset[0])
        self.assertEqual(mode_for_sample(42,2,row['id']),mode_for_sample(42,2,row['id']))
        counts={mode:sum(mode_for_sample(42,1,f'train/{i}')==mode for i in range(10000)) for mode in MODES}
        for mode,expected in zip(MODES,(2500,2000,2000,3500)):
            self.assertLess(abs(counts[mode]-expected),180)
    def test_alignment_includes_missing_rejects_missing_ids(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=root/'data'; media=root/'media'/'runs'; data.mkdir(parents=True)
            row=self.row(); row.pop('label')
            (data/'train.inputs.jsonl').write_text(json.dumps(row)+'\n')
            (data/'train.gold.jsonl').write_text(json.dumps({'id':row['id'],'label':'joy'})+'\n')
            for path,modality in zip(MEDIA_FILES['train'],('audio','video')):
                p=media/path; p.parent.mkdir(parents=True,exist_ok=True)
                rec={'sample_id':row['id'],'split':'train','modality':modality,'status':'missing_media','description':None}
                p.write_text(json.dumps(rec)+'\n')
            aligned=load_split(data,root/'media','train',True)
            self.assertEqual(aligned[0]['label'],'joy')
            self.assertIsNone(aligned[0]['audio_description'])
            (media/MEDIA_FILES['train'][0]).write_text('')
            with self.assertRaises(ValueError): load_split(data,root/'media','train',True)

if __name__=='__main__': unittest.main()
