import json
import tempfile
import unittest
from pathlib import Path
from visual_numbering import number_visuals


class VisualNumberingTests(unittest.TestCase):
    def test_gaps_letters_and_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            paths=[root/name for name in ['01_xg.png','05a_network.png','05b_network.png','60_loss.png']]
            for path in paths: path.write_bytes(path.name.encode())
            (root/'analysis_tables').mkdir()
            contract_file=root/'analysis_tables/chart_contracts.json'
            contract_file.write_text('{}')
            contracts={'60_loss.png': {'title':'Loss'}}
            result=number_visuals(paths,root,contracts)
            self.assertEqual([p.name for p in result],['01_xg.png','02_network.png','03_network.png','04_loss.png'])
            self.assertEqual(result[1].read_bytes(),b'05a_network.png')
            self.assertEqual(json.loads(contract_file.read_text()),{'04_loss.png':{'title':'Loss'}})
