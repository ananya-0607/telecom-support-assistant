"""Copy the bundled demo into empty storage, then start the API.

Existing active databases are never overwritten; no Groq calls are made.
"""
import json
import os
from pathlib import Path
import shutil


def seed_demo(root):
    destination = root/'data/indexes'
    pointer = destination/'search/active.json'
    if pointer.exists():
        return False
    bundle = root/'demo_data/indexes'
    active = json.loads((bundle/'search/active.json').read_text(encoding='utf-8'))
    build_id = active['build_id']
    if len(build_id) != 32 or any(c not in '0123456789abcdef' for c in build_id):
        raise ValueError('Invalid demo build ID')
    target = destination/'search/builds'/build_id
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle/'search/builds'/build_id, target, dirs_exist_ok=True)
    for item in bundle.iterdir():
        if item.is_file() and not (destination/item.name).exists():
            shutil.copy2(item, destination/item.name)
    # Publish last so an interrupted first-start copy can be retried.
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps(active, indent=2), encoding='utf-8')
    temporary.replace(pointer)
    return True


if __name__ == '__main__':
    if seed_demo(Path(__file__).resolve().parents[1]):
        print('Demo database loaded into persistent storage.', flush=True)
    os.execvp('python', ['python', '-m', 'uvicorn', 'telecom_support.api:app',
                        '--host', '0.0.0.0', '--port', '8000', '--workers', '1'])
