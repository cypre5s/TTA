"""Render only new public documents in the isolated expansion database."""
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
from scripts.process import process


def main():
    cv2.setNumThreads(1)
    pilot=sqlite3.connect('../../database/pilot.sqlite')
    known={r[0] for r in pilot.execute('select sha256 from documents')}
    db=sqlite3.connect('database/pilot.sqlite',timeout=45)
    with ProcessPoolExecutor(max_workers=3) as pool:
        while True:
            rows=db.execute('select sha256,path,pages from documents order by rowid').fetchall()
            pending=[r for r in rows if r[0] not in known and not Path('data/rendered',r[0],'manifest.json').exists()]
            if pending:
                for f in [pool.submit(process,*r) for r in pending]:f.result()
            elif len(rows)>=285 or Path('COLLECTION_DONE').exists():break
            else:time.sleep(12)

if __name__=='__main__':main()
