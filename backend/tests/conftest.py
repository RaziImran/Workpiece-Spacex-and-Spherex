import os
import tempfile

# Keep tests away from the real cache. Must happen before `app` is imported.
os.environ.setdefault("CACHE_DIR", tempfile.mkdtemp(prefix="spherex-test-cache-"))
