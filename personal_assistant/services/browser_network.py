"""Worker-only HTTP/CONNECT proxy. Resolve, validate, then connect to that IP.

Chromium uses this for HTTP(S), redirects, WebSockets and subresources. No proxy
bypass, QUIC or non-proxied WebRTC is enabled by the worker. This is a network
destination policy, not an OS sandbox for a compromised Chromium process.
"""
import asyncio
from contextlib import suppress
import ipaddress
import socket
from urllib.parse import urlsplit


class PublicProxy:
    def __init__(self, *, test_origin: tuple[str, int] | None = None):
        self.test_origin = test_origin
        self.server = None
        self.tasks = set()
        self.blocked = 0

    async def resolve(self, host: str, port: int) -> str:
        host = host.rstrip('.').lower()
        if self.test_origin == (host, port):
            return host
        if port not in (80, 443) or host == 'localhost' or host.endswith(('.localhost', '.local', '.internal', '.lan')):
            raise ValueError('Private networks and non-web ports are blocked')
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
            addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError('Private or mixed DNS destinations are blocked')
        return str(addresses[0])

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._serve, '127.0.0.1', 0, limit=32768)
        return f'http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}'

    async def _serve(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        remote = None
        try:
            async with asyncio.timeout(120):
                header = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 10)
                method, target, version = header.split(b'\r\n', 1)[0].decode('ascii').split(' ')
                connect = method == 'CONNECT'
                parsed = urlsplit(('https://' if connect else '') + target)
                if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError('Invalid destination')
                port = parsed.port or (443 if parsed.scheme == 'https' else 80)
                address = await self.resolve(parsed.hostname, port)
                upstream, remote = await asyncio.wait_for(asyncio.open_connection(address, port), 10)
                if connect:
                    writer.write(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                else:
                    path = parsed.path or '/'
                    if parsed.query:
                        path += '?' + parsed.query
                    lines = header.split(b'\r\n')[1:-2]
                    lines = [line for line in lines if line.split(b':', 1)[0].lower() not in
                             (b'proxy-connection', b'proxy-authorization', b'connection', b'host')]
                    lines += [b'Host: ' + parsed.netloc.encode('ascii'), b'Connection: close']
                    remote.write(f'{method} {path} {version}\r\n'.encode('ascii') + b'\r\n'.join(lines) + b'\r\n\r\n')
                    await remote.drain()
                await writer.drain()
                async def copy(source, destination):
                    while data := await source.read(65536):
                        destination.write(data)
                        await destination.drain()
                transfers = [asyncio.create_task(copy(reader, remote)), asyncio.create_task(copy(upstream, writer))]
                try:
                    await asyncio.wait(transfers, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for transfer in transfers:
                        transfer.cancel()
                    await asyncio.gather(*transfers, return_exceptions=True)
        except (ValueError, OSError, TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            self.blocked += 1
            with suppress(OSError):
                writer.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                await writer.drain()
        finally:
            if remote:
                remote.close()
            writer.close()
            self.tasks.discard(task)

    async def aclose(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
