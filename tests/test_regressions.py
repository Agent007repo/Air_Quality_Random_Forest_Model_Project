import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
ROOT = Path(__file__).resolve().parents[1]

def definitions(filename, names, namespace):
    path = ROOT / filename
    if path.suffix == '.ipynb':
        nb = json.loads(path.read_text())
        text = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        text = '\n'.join(line if not line.startswith(('!', '%', 'pip install')) else '# '+line for line in text.splitlines())
    else:
        text = path.read_text()
    tree = ast.parse(text)
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace
import pandas as pd
import numpy as np
import tempfile, os
from sklearn.impute import SimpleImputer

class AirQualityRegressionTests(unittest.TestCase):
    def test_preprocessing_is_chronological_and_train_only(self):
        nb=json.loads((ROOT/'Air_Quality_Random_Forest_Model_Samarth.ipynb').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'air.csv'
            count=40
            frame=pd.DataFrame({'Date':['01/01/2004']*20+['02/01/2004']*20,
                'Time':[f'{i%20:02d}.00.00' for i in range(count)],'CO(GT)':np.arange(count)/10+1,
                'PT08.S1(CO)':np.arange(count,dtype=float),'T':np.arange(count,dtype=float)+1,
                'RH':np.arange(count,dtype=float)+2,'AH':np.arange(count,dtype=float)+3})
            frame.loc[39,'T']=1e9
            frame.to_csv(path,sep=';',decimal=',',index=False)
            before=os.environ.get('AIR_QUALITY_CSV');os.environ['AIR_QUALITY_CSV']=str(path)
            try:
                namespace={};exec(''.join(nb['cells'][6]['source']),namespace)
                exec(''.join(nb['cells'][10]['source']),namespace)
                self.assertLess(namespace['X_train'].index.max(),namespace['X_test'].index.min())
                self.assertLess(namespace['imputer'].statistics_[2],100.)
                self.assertEqual(len(namespace['y_train']),32)
            finally:
                if before is None: os.environ.pop('AIR_QUALITY_CSV',None)
                else: os.environ['AIR_QUALITY_CSV']=before
