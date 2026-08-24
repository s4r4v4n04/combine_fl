import os
import socket

def get_ip_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

def get_ip_address_docker() -> str:
    return socket.gethostbyname(socket.gethostname())

def get_advertise_ip(local_ip: str) -> str:
    """Address the client tells the server to dial back on (its sync_port).

    By default this is the client's own primary IP, which only works when the
    server can route to it (same host / same or peered VNet). When the server is
    on a different network, that primary IP is a private address the server can't
    reach. Set FLO_ADVERTISE_IP to a server-reachable address (the VM's public IP,
    or a VPN/overlay IP such as Tailscale/WireGuard) to advertise that instead.
    The client still BINDS on all interfaces, so overriding this is safe."""
    return os.environ.get("FLO_ADVERTISE_IP") or local_ip
