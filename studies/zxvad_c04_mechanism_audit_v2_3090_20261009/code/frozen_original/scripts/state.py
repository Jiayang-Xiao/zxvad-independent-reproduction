"""Small stdlib-only state writer for setup, before PyTorch is installed."""
import json,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'outputs';out.mkdir(exist_ok=True)
value={'phase':sys.argv[1],'updated_unix':time.time()}
if len(sys.argv)>2:value['message']=' '.join(sys.argv[2:])
temp=out/'state.json.tmp';temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(out/'state.json')
