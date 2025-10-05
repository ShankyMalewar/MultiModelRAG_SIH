from importlib import import_module
import traceback, sys
try:
    mod = import_module('app.ingest.ingest')
    fn = getattr(mod, 'run_ingest', None) or getattr(mod, 'ingest_file', None)
    if not fn:
        print('NO ingest function found in app.ingest.ingest')
        sys.exit(1)
    print('Calling run_ingest synchronously ...')
    res = fn(r'C:\Users\malew\Desktop\All Projects\Multimodel RAG\data\vault\REPLACE_WITH_YOUR_SAVED_FILENAME.pdf')
    try:
        print('run_ingest returned:', getattr(res, 'to_dict', lambda: res)())
    except Exception:
        print('run_ingest returned object (no to_dict()). repr:')
        print(repr(res))
except Exception:
    traceback.print_exc()
