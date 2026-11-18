"""Absolute request timer nested inside the existing phase watchdog, without extending it."""
import contextlib,signal,threading,time
REQUEST_SECONDS=90.0
class RequestDeadline(BaseException):pass

@contextlib.contextmanager
def request_deadline(phase_deadline):
 if threading.current_thread() is not threading.main_thread():raise RuntimeError('request watchdog requires main thread')
 started=time.monotonic();remaining,interval=signal.getitimer(signal.ITIMER_REAL)
 if interval:raise RuntimeError('periodic alarm unsupported')
 handler=signal.getsignal(signal.SIGALRM)
 previous_expiry=started+remaining if remaining else None
 request_expiry=min(started+REQUEST_SECONDS,phase_deadline)
 phase_fired=False
 def expire(signum,frame):
  nonlocal phase_fired
  if previous_expiry is not None and previous_expiry<=request_expiry:
   phase_fired=True
   handler(signum,frame)
  raise RequestDeadline()
 try:
  signal.signal(signal.SIGALRM,expire)
  signal.setitimer(signal.ITIMER_REAL,max(.000001,min(request_expiry,previous_expiry or request_expiry)-time.monotonic()))
  yield request_expiry
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  signal.signal(signal.SIGALRM,handler)
  if previous_expiry is not None and not phase_fired:signal.setitimer(signal.ITIMER_REAL,max(.000001,previous_expiry-time.monotonic()))
