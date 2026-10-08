"""Immutable disjoint queues; physical GPU equals lane index."""
from model import ORDER
STUDY='ZXVAD_C04_SCREEN_DUALGPU_3090_20261008'
QUEUES={'0': ['B_s17', 'S25_s17', 'SR25_s17', 'C7_s17', 'R50_s17', 'G25_s17', 'CM7_s17', 'R100_s17', 'S12_s17', 'S50_s17', 'G00_s17'], '1': ['M20_s17', 'MR20_s17', 'U9_s17', 'QC03_s17', 'C3_s17', 'M10_s17', 'F9_s17', 'QC01_s17', 'C1_s17', 'M20W_s17', 'Tweak_s17']}
assert all(len(q)==11 for q in QUEUES.values())
assert len(set(sum(QUEUES.values(),[])))==22 and set(sum(QUEUES.values(),[]))==set(ORDER)
def lane_for(candidate):
    return next(lane for lane,q in QUEUES.items() if candidate in q)
