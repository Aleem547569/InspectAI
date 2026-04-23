import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
workers = 1
worker_class = "sync"
timeout = 120
max_requests = 100        # restart worker periodically to free memory
max_requests_jitter = 20
