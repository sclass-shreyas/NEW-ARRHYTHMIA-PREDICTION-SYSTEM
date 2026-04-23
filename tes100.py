import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from predict import predict_record

result = predict_record('100')
s = result['summary']
print(f"Record      : {s['record_id']}")
print(f"Total beats : {s['total_beats']}")
print(f"PVC flagged : {s['pvc_count']}")
print(f"Risk score  : {s['risk_score']:.4f}  ({s['risk_score']*100:.1f}% of beats)")