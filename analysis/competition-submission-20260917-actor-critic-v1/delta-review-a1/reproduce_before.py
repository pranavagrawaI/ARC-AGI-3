import importlib.util,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('prior_defect_fixture',R/'independent-review/qualify_gateway_deep.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.PACKAGE=R/'checks/repair-a1/before-package'
sys.argv=[str(R/'independent-review/qualify_gateway_deep.py'),'--games','2','--mode','deep','--output',str(O/'deep-before-reproduced')]
m.main()
