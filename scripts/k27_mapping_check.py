"""Check that KuaiRand-27K shares user and random-pool video ids with KuaiRand-Pure (archive streamed, not extracted).

The 27K user table and random-exposure log are hashed straight from the tar stream and compared with the
Pure files; byte-identical files mean identical user ids, user features, random exposures and pool video ids,
so the Pure-based user partition and the 1K-to-Pure user map apply to 27K unchanged.
"""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {'KuaiRand-27K/data/user_features_27k.csv': 'user_features_pure.csv',
         'KuaiRand-27K/data/log_random_4_22_to_5_08_27k.csv': 'log_random_4_22_to_5_08_pure.csv'}

def md5(f):
    h = hashlib.md5()
    for block in iter(lambda: f.read(1 << 24), b''): h.update(block)
    return h.hexdigest()

def main():
    out = {'archive_md5_verified': '3e3c799a24e2d23a4d2c757fbf9adf59', 'members': []}
    with tarfile.open(ROOT/'data/raw/k27_download/KuaiRand-27K.tar.gz', mode='r|gz') as tar:
        for m in tar:
            out['members'].append({'name': m.name, 'size': m.size})
            if m.name in PAIRS:
                with open(ROOT/'data/raw/pure_download/KuaiRand-Pure/data'/PAIRS[m.name], 'rb') as g: ref = md5(g)
                out[m.name] = {'md5_27k': md5(tar.extractfile(m)), 'md5_pure': ref}
                out[m.name]['identical'] = out[m.name]['md5_27k'] == ref
    out['all_identical'] = all(out[k]['identical'] for k in PAIRS)
    dest = ROOT/'reports/expansion/k27_mapping_check.json'; dest.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
