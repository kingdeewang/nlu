from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters
import os
from util import utility
import _thread
import time

# gateway = JavaGateway()

# res = os.popen('tasklist | findstr "java"').readlines()
#
# pid = -1
# for line in res:
#     print(line)
#     line = line.split()
#     print(line)
#     if len(line) >= 2:
#         pid = line[1]
#         break
#
# if pid >= 0:
#     os.system('taskkill -PID ' + pid + ' -F')


def classpath(libpath=utility.workingDirectory + 'Chatbot/target/lib'):
    classpath = []
    for name in os.listdir(libpath):
        path = os.path.join(libpath , name)

        if path.endswith('jar'):
            classpath.append(path)
    return classpath


classpaths = classpath(utility.workingDirectory + 'Chatbot/target/lib')
classpaths.append(utility.workingDirectory + 'Chatbot/target/classes')

cmd = "java -Dfile.encoding=UTF-8 -ea -Xms1g -Xmx1g -classpath " + ';'.join(classpaths) + " com.util.EntryPoint"
print(cmd[:100] + '......' + cmd[-100:])

_thread.start_new_thread(os.system, (cmd,))

gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True, auto_field=True),
                      callback_server_parameters=CallbackServerParameters())

while True:
    try:
        jvm = gateway.jvm
        com = jvm.com
        break
    except:
        seconds = 2
        time.sleep(seconds)
        print('sleeping for %d second(s)' % seconds)


class Callback():

    def notify(self, s):
        print(s)

    class Java:
        implements = ["com.util.EntryPoint$Callback"]


# py4j.java_gateway.set_field(SyntacticTree(-1) , 'pythonCallback' , Callback())
gateway.entry_point.register(Callback())


def test_jpype():
#     https://www.lfd.uci.edu/~gohlke/pythonlibs/#jpype
#     http://www.kumartarun.com/comparison_of_libraries.html
#     https://github.com/ninia/jep
#     https://jpype.readthedocs.io/en/latest/index.html
#     https://pyjnius.readthedocs.io/en/stable/index.html
    import jpype
    from jpype import java, JClass, JPackage

    classpath = []
    classpath.append(utility.workingDirectory + 'Chatbot/target/classes')

    libpath = utility.workingDirectory + 'Chatbot/target/lib/'
    for name in os.listdir(libpath):
        path = os.path.join(libpath , name)

        if path.endswith('jar'):
            classpath.append(path)

    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", '-Xms1g', '-Xmx1g', '-Dfile.encoding=UTF-8', "-Djava.class.path=" + ';'.join(classpath))

    com = JPackage('com')
    java.lang.System.out.println("hello world")

    EntryPoint = com.util.EntryPoint

    EntryPoint.test([])

    jpype.shutdownJVM()


if __name__ == '__main__':
    ...
