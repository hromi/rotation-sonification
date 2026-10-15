#!/usr/bin/env python3
"""
Start (or restart) the HL2SS server app on one or more HoloLens 2 devices
through the Windows Device Portal REST API, then wait until the Spatial
Input port (3812) accepts connections.

    python hl2ss_launch.py 192.168.1.102 192.168.1.150 --restart

Credentials: --user/--password, or env WDP_USER / WDP_PASSWORD,
otherwise prompted.
"""
import argparse
import base64
import getpass
import os
import socket
import sys
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SPATIAL_INPUT_PORT = 3812


def b64(s):
    return base64.b64encode(s.encode()).decode()


class DevicePortal:
    def __init__(self, host, user, password):
        self.base = f"https://{host}"
        self.s = requests.Session()
        self.s.auth = (user, password)
        self.s.verify = False  # HoloLens uses a self-signed certificate
        r = self.s.get(self.base + "/api/os/machinename", timeout=10)
        r.raise_for_status()
        self.name = r.json().get("ComputerName", host)
        token = self.s.cookies.get("CSRF-Token")
        if token:
            self.s.headers["X-CSRF-Token"] = token

    def find_package(self, needle="hl2ss"):
        r = self.s.get(self.base + "/api/app/packagemanager/packages", timeout=10)
        r.raise_for_status()
        for p in r.json()["InstalledPackages"]:
            if needle in p["Name"].lower():
                return p
        raise RuntimeError(f"{self.name}: no installed package matching '{needle}'")

    def stop(self, pkg):
        self.s.delete(self.base + "/api/taskmanager/app",
                      params={"package": b64(pkg["PackageFullName"]), "forcestop": "yes"},
                      timeout=10)

    def start(self, pkg, attempts=4):
        # Right after a stop the portal often answers 500; retrying succeeds.
        for i in range(attempts):
            r = self.s.post(self.base + "/api/taskmanager/app",
                            params={"appid": b64(pkg["PackageRelativeId"]),
                                    "package": b64(pkg["PackageFullName"])},
                            timeout=15)
            if r.ok or i == attempts - 1:
                r.raise_for_status()
                return
            time.sleep(3)


def wait_for_stream(host, timeout=30.0):
    """Wait until HL2SS actually delivers Spatial Input bytes (not just accepts TCP)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, SPATIAL_INPUT_PORT), timeout=3) as s:
                s.settimeout(3)
                if s.recv(64):
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hosts", nargs="+")
    ap.add_argument("--user", default=os.environ.get("WDP_USER"))
    ap.add_argument("--password", default=os.environ.get("WDP_PASSWORD"))
    ap.add_argument("--restart", action="store_true", help="stop HL2SS first (wakes a suspended instance)")
    args = ap.parse_args()

    user = args.user or input("Device Portal user: ")
    password = args.password or getpass.getpass("Device Portal password: ")

    ok = True
    for host in args.hosts:
        try:
            dp = DevicePortal(host, user, password)
            pkg = dp.find_package()
            if args.restart:
                dp.stop(pkg)
                time.sleep(2)
            dp.start(pkg)
            print(f"[{host}] {dp.name}: HL2SS launched, waiting for Spatial Input stream...", flush=True)
            if wait_for_stream(host):
                print(f"[{host}] streaming OK")
            else:
                ok = False
                print(f"[{host}] no data on port {SPATIAL_INPUT_PORT} "
                      "(is the device awake and worn? HL2SS needs to be the foreground app)")
        except Exception as e:
            ok = False
            print(f"[{host}] FAILED: {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
