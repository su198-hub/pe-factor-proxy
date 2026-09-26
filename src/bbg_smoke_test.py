"""Check that the Bloomberg Desktop API is reachable and returns data.

Run from the project root:  .venv\\Scripts\\python.exe -u src\\bbg_smoke_test.py
"""
import blpapi

HOST, PORT = "localhost", 8194


def main():
    opts = blpapi.SessionOptions()
    opts.setServerHost(HOST)
    opts.setServerPort(PORT)
    opts.setConnectTimeout(5_000)
    session = blpapi.Session(opts)

    print("starting session...", flush=True)
    if not session.start():
        raise SystemExit("Could not start session - is the terminal running and logged in?")
    print("session started; opening //blp/refdata...", flush=True)
    if not session.openService("//blp/refdata"):
        raise SystemExit("Could not open //blp/refdata")
    svc = session.getService("//blp/refdata")

    req = svc.createRequest("HistoricalDataRequest")
    req.getElement("securities").appendValue("SPX Index")
    req.getElement("fields").appendValue("PX_LAST")
    req.set("startDate", "20250901")
    req.set("endDate", "20250930")
    req.set("periodicitySelection", "WEEKLY")
    print("service open; sending request...", flush=True)
    session.sendRequest(req)

    while True:
        ev = session.nextEvent(10_000)
        for msg in ev:
            if msg.hasElement("responseError"):
                print("ERROR:", msg.getElement("responseError"))
            elif msg.hasElement("securityData"):
                sd = msg.getElement("securityData")
                if sd.hasElement("securityError"):
                    print("SECURITY ERROR:", sd.getElement("securityError"))
                fd = sd.getElement("fieldData")
                for i in range(fd.numValues()):
                    row = fd.getValueAsElement(i)
                    print(row.getElementAsDatetime("date"), row.getElementAsFloat("PX_LAST"))
        if ev.eventType() == blpapi.Event.RESPONSE:
            break
        if ev.eventType() == blpapi.Event.TIMEOUT:
            print("Timed out waiting for response")
            break
    session.stop()


if __name__ == "__main__":
    main()
