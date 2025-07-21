# sentenial_service.py

import win32serviceutil
import win32service
import win32event
import servicemanager
import os
import sys
import time

class SentenialXThreatMonitor(win32serviceutil.ServiceFramework):
    _svc_name_ = "SentenialXThreatMonitor"
    _svc_display_name_ = "Sentenial-X Threat Monitor Service"
    _svc_description_ = "Real-time threat detection engine by Sentenial-X"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.running = False

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.main()

    def main(self):
        script_path = os.path.join(os.path.dirname(__file__), "threat_monitor.py")
        while self.running:
            os.system(f"python {script_path}")
            time.sleep(10)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(SentenialXThreatMonitor)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(SentenialXThreatMonitor)