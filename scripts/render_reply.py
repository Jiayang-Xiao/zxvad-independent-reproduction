"""Generate a copy-ready email only after verifying the public pushed commit."""
import argparse
import json
import subprocess
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',default='Jiayang-Xiao/zxvad-independent-reproduction')
    ap.add_argument('--output',default=str(ROOT/'.private/author_reply_ready.txt'))
    ap.add_argument('--skip-network',action='store_true',help='Local testing only; draft is labeled unverified')
    ap.add_argument('--commit',help='Local rendering test override, requires --skip-network')
    args=ap.parse_args()
    if args.commit and not args.skip_network: raise RuntimeError('Commit override is for offline testing only')
    commit=args.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit): raise RuntimeError('Invalid commit')
    if len(args.repo.split('/'))!=2 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./' for c in args.repo): raise RuntimeError('Invalid repository')
    url='https://github.com/'+args.repo
    if not args.skip_network:
        for endpoint in (f'https://api.github.com/repos/{args.repo}',f'https://api.github.com/repos/{args.repo}/commits/{commit}'):
            req=urllib.request.Request(endpoint,headers={'User-Agent':'zxvad-independent-reproduction','Accept':'application/vnd.github+json'})
            with urllib.request.urlopen(req,timeout=45) as response: data=json.load(response)
            if '/commits/' in endpoint:
                if data['sha']!=commit: raise RuntimeError('Public commit mismatch')
            elif data.get('private') is not False: raise RuntimeError('Repository is not publicly readable')
    replacements={'REPO_URL':url,'COMMIT_URL':url+'/tree/'+commit,
        'REVIEW_URL':url+'/blob/'+commit+'/docs/author_review.md','MODEL_URL':url+'/blob/'+commit+'/src/baseline_zxvad.py',
        'TRAIN_URL':url+'/blob/'+commit+'/src/train.py','EVAL_URL':url+'/blob/'+commit+'/src/evaluate.py'}
    text=(ROOT/'docs/author_reply_template.txt').read_text(encoding='utf-8')
    for key,value in replacements.items(): text=text.replace('@@'+key+'@@',value)
    if '@@' in text: raise RuntimeError('Unfilled reply placeholder')
    if args.skip_network: text='[LOCAL DRAFT; PUBLICATION NOT VERIFIED]\n\n'+text
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding='utf-8')
    print('Copy-ready reply: '+str(path))

if __name__=='__main__': main()
