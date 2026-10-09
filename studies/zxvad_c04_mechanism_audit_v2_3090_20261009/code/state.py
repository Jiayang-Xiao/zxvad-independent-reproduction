import sys,time
from pathlib import Path
from study_support import write
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    write(ROOT/'outputs/state.json',{'phase':sys.argv[1],'updated_unix':time.time(),'message':sys.argv[2] if len(sys.argv)>2 else ''})
