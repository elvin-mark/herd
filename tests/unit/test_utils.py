import socket

from herd.services.manager import find_free_port


def test_find_free_port():
    """Verifies that find_free_port returns a valid, bindable TCP port number."""
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535

    # Verify port can be bound
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.close()
