"""Thin helpers over the Bloomberg Desktop API.

xbbg handles bdh/bdp; this module adds ticker search, which xbbg does not expose.
The terminal must be running and logged in on this machine.
"""
import blpapi

HOST, PORT = "localhost", 8194


def _session(service):
    opts = blpapi.SessionOptions()
    opts.setServerHost(HOST)
    opts.setServerPort(PORT)
    opts.setConnectTimeout(5_000)
    s = blpapi.Session(opts)
    if not s.start():
        raise RuntimeError("Could not start Bloomberg session - is the terminal running and logged in?")
    if not s.openService(service):
        s.stop()
        raise RuntimeError(f"Could not open {service}")
    return s


def search(queries, yk_filter="YK_FILTER_INDX", max_results=50):
    """Search the instruments service. Returns a list of (query, security, description).

    yk_filter: YK_FILTER_INDX, YK_FILTER_EQTY, YK_FILTER_NONE, ...
    """
    if isinstance(queries, str):
        queries = [queries]
    s = _session("//blp/instruments")
    svc = s.getService("//blp/instruments")
    out = []
    try:
        for q in queries:
            req = svc.createRequest("instrumentListRequest")
            req.set("query", q)
            req.set("yellowKeyFilter", yk_filter)
            req.set("maxResults", max_results)
            s.sendRequest(req)
            while True:
                ev = s.nextEvent(10_000)
                for msg in ev:
                    if msg.hasElement("results"):
                        res = msg.getElement("results")
                        for i in range(res.numValues()):
                            r = res.getValueAsElement(i)
                            sec = r.getElementAsString("security").replace("<index>", " Index")
                            out.append((q, sec, r.getElementAsString("description")))
                if ev.eventType() in (blpapi.Event.RESPONSE, blpapi.Event.TIMEOUT):
                    break
    finally:
        s.stop()
    return out
