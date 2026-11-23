import contextlib
import subprocess

@contextlib.contextmanager
def listener(command, scheme):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        prefix = "Serving " + scheme + "://127.0.0.1:"
        for line in proc.stdout:
            if line.startswith(prefix):
                break
        else:
            raise AssertionError((command, proc.poll()))
        yield int(line[len(prefix):])
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()
