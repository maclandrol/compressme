import json
import pathlib
import time
import psutil

processes=[]
for process in psutil.process_iter():
    try:
        process.cpu_percent(None)
        processes.append(process)
    except (psutil.AccessDenied,psutil.NoSuchProcess):
        pass
time.sleep(1)
rows=[]
for process in processes:
    try:
        rows.append({'name':process.name(),'cpu_percent':process.cpu_percent(None),
                     'rss_mb':round(process.memory_info().rss/1e6)})
    except (psutil.AccessDenied,psutil.NoSuchProcess):
        pass
report={'memory':psutil.virtual_memory()._asdict(),'swap':psutil.swap_memory()._asdict(),
        'processes':sorted(rows,key=lambda x:x['cpu_percent'],reverse=True)[:10]}
pathlib.Path(__file__).with_name('host-observation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
