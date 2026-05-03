import wfdb
import os

INCART_DIR = os.path.dirname(os.path.abspath(__file__))

for record_id in ["I01", "I04"]:
    record = wfdb.rdrecord(os.path.join(INCART_DIR, record_id))
    ann    = wfdb.rdann(os.path.join(INCART_DIR, record_id), "atr")
    
    print(f"\nRecord: {record_id}")
    print(f"  Signal shape : {record.p_signal.shape}")
    print(f"  Sample rate  : {record.fs} Hz")
    print(f"  Leads        : {record.sig_name}")
    print(f"  Total beats  : {len(ann.sample)}")
    print(f"  Unique symbols: {set(ann.symbol)}")