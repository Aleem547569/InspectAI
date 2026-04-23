import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
workers = 1
worker_class = "gthread"
threads = 4
timeout = 120
