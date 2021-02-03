
import os
import re
import sys
from util.utility import get_process_pid
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime, timedelta
import requests
from util import utility


def task(url):
    print('now is the time =', datetime.now())
    print('training url = ', url)
    requests.post(url)


# https://apscheduler.readthedocs.io/en/latest/index.html
# https://apscheduler.readthedocs.io/en/latest/modules/triggers/interval.html
def start_schedule(tcp):
    print('os.getpid() =', os.getpid())

#     cmd = 'ps aux|grep python|grep -v grep|grep gunicorn.py|grep tcp=%s|cut -c 9-15|xargs kill -15' % tcp
    cmd = 'ps aux|grep python|grep -v grep|grep gunicorn.py|grep tcp=%s|cut -c 9-15' % tcp
    print(cmd)
    res = os.popen(cmd).readlines()
    res = [s.strip() for s in res]
    print(res)

    del res[res.index(str(os.getpid()))]

    for pid in res:
        print('kill -9 %s' % pid)
        os.system('kill -9 %s' % pid)

    start_date = datetime.now()
    print('present time =', start_date)
    start_date -= timedelta(hours=start_date.hour, minutes=start_date.minute, seconds=start_date.second, microseconds=start_date.microsecond)
    # start the task the 2:00am at the next date
    start_date += timedelta(days=1, hours=2)
    print('start date =', start_date)
    print('start_schedule')

    import daemon
    with daemon.DaemonContext():

        log = logging.getLogger('apscheduler.executors.default')
        log.setLevel(logging.INFO)  # DEBUG

        fmt = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        log.addHandler(h)

        scheduler = BlockingScheduler()

        scheduler.add_job(func=task,
                          id='training',
                          args=('http://localhost:%d/schedule' % tcp,),
                          trigger='interval',  # periodically run the task
                          days=1,
                          start_date=start_date,
                          replace_existing=True)
        scheduler.start()


# usage:
# python3 gunicorn.py --tcp=2000 --clean_gpu=0 --workers=4 --debug --worker-class=gevent
# python3 gunicorn.py --tcp=2000 --workers=4 --debug --worker-class=gevent
# python3 gunicorn.py --tcp=8000 --interface=nlp_app --debug --schedule --timeout
if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    if len(sys.argv) < 1:
        raise Exception('tcp number is missing!')

    clean_gpu = -1

    num_workers = 1
    debug = False
    schedule = False
    stop = False
    worker_class = 'sync'
    interface = 'interface'
    timeout = 86400
    for s in sys.argv:
        m = re.compile('--tcp=(\d+)').fullmatch(s)
        if m:
            tcp = int(m.group(1))
            continue
        m = re.compile('--clean_gpu=(\d+)').fullmatch(s)
        if m:
            clean_gpu = int(m.group(1))
            continue
        m = re.compile('--workers=(\d+)').fullmatch(s)
        if m:
            num_workers = int(m.group(1))
            continue

        m = re.compile('--debug').fullmatch(s)
        if m:
            debug = True
            continue

        m = re.compile('--schedule').fullmatch(s)
        if m:
            schedule = True
            continue

        m = re.compile('--stop').fullmatch(s)
        if m:
            stop = True
            continue

        m = re.compile('--timeout=(\d+)').fullmatch(s)
        if m:
            timeout = int(m.group(1))
            continue

        m = re.compile('--worker-class=(\w+)').fullmatch(s)
        if m:
            worker_class = m.group(1)
            continue

        m = re.compile('--interface=(\w+)').fullmatch(s)
        if m:
            interface = m.group(1)
            continue

    for pid in get_process_pid(tcp):
        os.system('kill -9 ' + pid)

    cwd = os.getcwd()

    accesslog = os.path.join(cwd, '../log/access%d.txt' % tcp)
    errorlog = os.path.join(cwd, '../log/error%d.txt' % tcp)

    if not os.path.isfile(accesslog):
        print('%s does not exist, create a new txt file' % accesslog)
        utility.createNewFile(accesslog)
    elif os.path.getsize(accesslog) // 1024 // 1024 > 30:
        with open(accesslog, 'w') as _:
            ...

    if not os.path.isfile(errorlog):
        print('%s does not exist, create a new txt file' % errorlog)
        utility.createNewFile(errorlog)
    elif os.path.getsize(errorlog) // 1024 // 1024 > 30:
        with open(errorlog, 'w') as _:
            ...

    print('accesslog =', accesslog)
    print('errorlog =', errorlog)

    cmd = 'gunicorn --workers=%d '\
    '--bind=0.0.0.0:%d '\
    '--daemon '\
    '--timeout=%d '\
    '--access-logfile=%s '\
    '--error-logfile=%s '\
    '--log-level=debug '\
    '--capture-output '\
    '--worker-class=%s '\
    '%s:app' % (num_workers, tcp, timeout, accesslog, errorlog, worker_class, interface)

    print(cmd)

#     res = os.popen(cmd).readlines()
#     for s in res:
#         print(s)

    if clean_gpu >= 0:
        res = os.popen('nvidia-smi').readlines()
        setPID = set()
        for s in res:
# Displayed as "C" for Compute Process, "G" for Graphics Process, and "C+G" for the process having both Compute and Graphics contexts.
            m = re.compile('\| +(\d+) +(\d+) +(\S+) +(\S+) +(\d+MiB) +\|').match(s)  # can not use fullmatch because these is an '\n' at the end of each line!
            if m:
                GPU = m.group(1)
                PID = m.group(2)
                Type = m.group(3)
                ProcessName = m.group(4)
                Usage = m.group(5)

                if Type == 'C' and int(GPU) == clean_gpu:
                    print('GPU = %s, PID = %s, Type = %s, ProcessName = %s, Usage = %s' % (GPU , PID , Type, ProcessName, Usage))
                    setPID.add(PID)
#             else:
#                 print(s)
#                 print('does not match!')
        for pid in setPID:
            sudoPassword = '123456'
            command = 'sudo kill -9 ' + pid
            os.system('echo %s|sudo -S %s' % (sudoPassword, command))
#             os.system()

    if stop:
        print('gunicorn is stopped!')
        os.system('ps aux|grep python|grep -v grep|grep gunicorn.py|grep tcp=%s|cut -c 9-15|xargs kill -15' % tcp)
    else:
        os.system(cmd)

        if debug:
            os.system('tail -100f %s' % errorlog)

        if schedule:
            start_schedule(tcp)

"""
import os

workers = 1

bind = '0.0.0.0:2000'

daemon = True

timeout = 864000

cwd = os.getcwd()
accesslog = os.path.join(cwd, 'log/access.txt')
errorlog = os.path.join(cwd, 'log/error.txt')

debug = True

#pidfile = os.path.join(cwd, 'interface.pid')

logfile = os.path.join(cwd, 'log/log.txt')

loglevel = 'debug'

capture_output = True

"""

"""
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --log-syslog          Send *Gunicorn* logs to syslog. [False]
  --max-requests-jitter INT
                        The maximum jitter to add to the *max_requests*
                        setting. [0]
  --limit-request-fields INT
                        Limit the number of HTTP headers fields in a request.
                        [100]
  --reuse-port          Set the ``SO_REUSEPORT`` flag on the listening socket.
                        [False]
  -D, --daemon          Daemonize the Gunicorn process. [False]
  --proxy-protocol      Enable detect PROXY protocol (PROXY mode). [False]
  -w INT, --workers INT
                        The number of worker processes for handling requests.
                        [1]
  -u USER, --user USER  Switch worker processes to run as this user. [1004]
  --check-config        Check the configuration. [False]
  --reload-extra-file FILES
                        Extends :ref:`reload` option to also watch and reload
                        on additional files [[]]
  --backlog INT         The maximum number of pending connections. [2048]
  --reload              Restart workers when code changes. [False]
  --disable-redirect-access-to-syslog
                        Disable redirect access logs to syslog. [False]
  --error-logfile FILE, --log-file FILE
                        The Error log file to write to. [-]
  --pythonpath STRING   A comma-separated list of directories to add to the
                        Python path. [None]
  --proxy-allow-from PROXY_ALLOW_IPS
                        Front-end's IPs from which allowed accept proxy
                        requests (comma separate). [127.0.0.1]
  --access-logfile FILE
                        The Access log file to write to. [None]
  --log-config FILE     The log config file to use. [None]
  --log-config-dict LOGCONFIG_DICT
                        The log config dictionary to use, using the standard
                        Python [{}]
  --log-syslog-facility SYSLOG_FACILITY
                        Syslog facility name [user]
  --statsd-host STATSD_ADDR
                        ``host:port`` of the statsd server to log to. [None]
  --max-requests INT    The maximum number of requests a worker will process
                        before restarting. [0]
  --keep-alive INT      The number of seconds to wait for requests on a Keep-
                        Alive connection. [2]
  --certfile FILE       SSL certificate file [None]
  --preload             Load application code before the worker processes are
                        forked. [False]
  --paste STRING, --paster STRING
                        Load a PasteDeploy config file. The argument may
                        contain a ``#`` [None]
  --paste-global CONF   Set a PasteDeploy global config variable in
                        ``key=value`` form. [[]]
  --reload-engine STRING
                        The implementation that should be used to power
                        :ref:`reload`. [auto]
  -n STRING, --name STRING
                        A base to use with setproctitle for process naming.
                        [None]
  --suppress-ragged-eofs
                        Suppress ragged EOFs (see stdlib ssl module's) [True]
  --do-handshake-on-connect
                        Whether to perform SSL handshake on socket connect
                        (see stdlib ssl module's) [False]
  -g GROUP, --group GROUP
                        Switch worker process to run as this group. [1004]
  -m INT, --umask INT   A bit mask for the file mode on files written by
                        Gunicorn. [0]
  --cert-reqs CERT_REQS
                        Whether client certificate is required (see stdlib ssl
                        module's) [0]
  --limit-request-field_size INT
                        Limit the allowed size of an HTTP request header
                        field. [8190]
  --keyfile FILE        SSL key file [None]
  -c CONFIG, --config CONFIG
                        The Gunicorn config file. [None]
  --ciphers CIPHERS     Ciphers to use (see stdlib ssl module's) [TLSv1]
  -e ENV, --env ENV     Set environment variable (key=value). [[]]
  --log-syslog-prefix SYSLOG_PREFIX
                        Makes Gunicorn use the parameter as program-name in
                        the syslog entries. [None]
  --statsd-prefix STATSD_PREFIX
                        Prefix to use when emitting statsd metrics (a trailing
                        ``.`` is added, []
  -t INT, --timeout INT
                        Workers silent for more than this many seconds are
                        killed and restarted. [30]
  -R, --enable-stdio-inheritance
                        Enable stdio inheritance. [False]
  --worker-connections INT
                        The maximum number of simultaneous clients. [1000]
  -p FILE, --pid FILE   A filename to use for the PID file. [None]
  --log-level LEVEL     The granularity of Error log outputs. [info]
  --log-syslog-to SYSLOG_ADDR
                        Address to send syslog messages. [udp://localhost:514]
  -k STRING, --worker-class STRING
                        The type of workers to use. [sync]
  --graceful-timeout INT
                        Timeout for graceful workers restart. [30]
  --capture-output      Redirect stdout/stderr to specified file in
                        :ref:`errorlog`. [False]
  --no-sendfile         Disables the use of ``sendfile()``. [None]
  --chdir CHDIR         Chdir to specified directory before apps loading.
                        [/home/zlz/solution]
  --ssl-version SSL_VERSION
                        SSL version to use (see stdlib ssl module's)
                        [_SSLMethod.PROTOCOL_TLS]
  --ca-certs FILE       CA certificates file [None]
  --forwarded-allow-ips STRING
                        Front-end's IPs from which allowed to handle set
                        secure headers. [127.0.0.1]
  --spew                Install a trace function that spews every line
                        executed by the server. [False]
  --access-logformat STRING
                        The access log format. [%(h)s %(l)s %(u)s %(t)s
                        "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"]
  --worker-tmp-dir DIR  A directory to use for the worker heartbeat temporary
                        file. [None]
  -b ADDRESS, --bind ADDRESS
                        The socket to bind. [['127.0.0.1:8000']]
  --limit-request-line INT
                        The maximum size of HTTP request line in bytes. [4094]
  --threads INT         The number of worker threads for handling requests.
                        [1]
  --logger-class STRING
                        The logger you want to use to log events in Gunicorn.
                        [gunicorn.glogging.Logger]
  --initgroups          If true, set the worker process's group access list
                        with all of the [False]
                        
"""
