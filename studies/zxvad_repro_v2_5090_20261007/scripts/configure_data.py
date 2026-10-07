"""Set data locations before the frozen prepare record exists."""
import argparse,json,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--source');args=ap.parse_args()
base=Path(args.data_root).expanduser().resolve()
source=Path(args.source).expanduser().resolve() if args.source else base/('shanghai' if (base/'shanghai').exists() else 'shanghaitech')/'training'/'frames'
value={'source_frames':str(source),'targets':{t:{'frames':str(base/t/'testing'/'frames'),'labels':str(base/f'frame_labels_{t}.npy')} for t in ('ped1','ped2','avenue')}}
path=root/'data_config.json'
if (root/'outputs'/'prepared.json').exists() and (not path.exists() or json.loads(path.read_text())!=value):raise SystemExit('Data already frozen. Use a new workspace for changed data.')
path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value,indent=2))
