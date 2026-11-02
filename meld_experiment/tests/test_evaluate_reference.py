import csv
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from evaluate_meld import evaluate_meld


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.ref=[{'id':'dev/a','label':'joy'},{'id':'dev/b','label':'neutral'}]
    def tearDown(self): self.temp.cleanup()
    def write(self, rows):
        p=self.root/'p.csv'
        with p.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['sample_id','true_label','pred_label']); w.writeheader(); w.writerows(rows)
        return p
    def test_valid_invalid_predictions_count_as_errors(self):
        p=self.write([{'sample_id':'dev/a','true_label':'joy','pred_label':'oops'},
                      {'sample_id':'dev/b','true_label':'neutral','pred_label':'neutral'}])
        result=evaluate_meld(p,self.root/'out',expected_count=2,reference_rows=self.ref)
        self.assertEqual(result['accuracy'],0.5)
        self.assertEqual(result['weighted_f1'],0.5)
        self.assertEqual(result['invalid_count'],1)
    def test_same_count_wrong_id_and_wrong_gold_are_rejected(self):
        rows=[{'sample_id':'dev/a','true_label':'joy','pred_label':'joy'},
              {'sample_id':'dev/c','true_label':'neutral','pred_label':'neutral'}]
        with self.assertRaisesRegex(ValueError,'ID mismatch'):
            evaluate_meld(self.write(rows),self.root/'out',expected_count=2,reference_rows=self.ref)
        rows[1]['sample_id']='dev/b'; rows[1]['true_label']='anger'
        with self.assertRaisesRegex(ValueError,'True label mismatch'):
            evaluate_meld(self.write(rows),self.root/'out',expected_count=2,reference_rows=self.ref)
    def test_reference_and_unique_nonempty_ids_required(self):
        rows=[{'sample_id':'dev/a','true_label':'joy','pred_label':'joy'},
              {'sample_id':'dev/a','true_label':'neutral','pred_label':'neutral'}]
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            evaluate_meld(self.write(rows),self.root/'out',reference_rows=self.ref)
        rows[1]['sample_id']=''
        with self.assertRaisesRegex(ValueError,'Empty sample_id'):
            evaluate_meld(self.write(rows),self.root/'out',reference_rows=self.ref)
        with self.assertRaisesRegex(ValueError,'independent reference'):
            evaluate_meld(self.write(rows),self.root/'out')

if __name__=='__main__': unittest.main()
