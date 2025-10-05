import os, sys, time, traceback
from importlib import import_module

def main(path):
    t0 = time.time()
    print("DEBUG_CHILD: start", flush=True)
    print("DEBUG_CHILD: sys.executable =", sys.executable, flush=True)
    print("DEBUG_CHILD: sys.version   =", sys.version.replace("\n"," "), flush=True)
    print("DEBUG_CHILD: cwd           =", os.getcwd(), flush=True)
    print("DEBUG_CHILD: path arg      =", path, flush=True)
    print("DEBUG_CHILD: sys.path[0:3] =", sys.path[0:3], flush=True)

    try:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        print("DEBUG_CHILD: importing app.ingest.ingest ...", flush=True)
        t_import = time.time()
        m = import_module("app.ingest.ingest")
        print("DEBUG_CHILD: import OK in %.3fs" % (time.time()-t_import), flush=True)

        fn = getattr(m, "ingest_file", None) or getattr(m, "run_ingest", None)
        if not fn:
            print("DEBUG_CHILD: ERROR - ingest function not found on module", flush=True)
            sys.exit(2)

        print("DEBUG_CHILD: calling ingest function ->", path, flush=True)
        t_call = time.time()
        res = fn(path)
        dt = time.time()-t_call
        # Try to render result
        to_dict = getattr(res, "to_dict", None)
        rendered = to_dict() if callable(to_dict) else res
        print("DEBUG_CHILD: result:", rendered, flush=True)
        print("DEBUG_CHILD: ingest call returned in %.3fs" % dt, flush=True)
        rc = 0
    except SystemExit as e:
        rc = int(getattr(e, "code", 1) or 0)
        print("DEBUG_CHILD: SystemExit with code", rc, flush=True)
    except Exception:
        traceback.print_exc()
        rc = 3
    finally:
        print("DEBUG_CHILD: end, elapsed %.3fs" % (time.time()-t0), flush=True)
        sys.exit(rc)

if __name__ == "__main__":
    main(r"C:\Users\malew\Desktop\All Projects\Multimodel RAG\data\vault\REPLACE_WITH_YOUR_SAVED_FILENAME.pdf")
