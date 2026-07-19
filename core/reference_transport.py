from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import aiohttp
from aiohttp.connector import Connection

MAX_REDIRECTS = 4
MAX_URL_LENGTH = 8192
READ_CHUNK_SIZE = 64 * 1024


class Resolver(Protocol):
    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[dict[str, str | int]]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RemoteReferenceDenied(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class RemoteLimitExceeded(Exception):
    max_bytes: int

    def __str__(self) -> str:
        return f"remote reference exceeded {self.max_bytes} bytes"


@dataclass(frozen=True, slots=True)
class _UnsafeRemoteAddress(OSError):
    host: str

    def __str__(self) -> str:
        return f"remote host must resolve only to public addresses: {self.host}"


@dataclass(frozen=True, slots=True)
class _RemoteTarget:
    url: str
    host: str
    port: int


def _public_ip(value: str) -> str | None:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return None
    return address.compressed if address.is_global else None


def _target(url: str) -> _RemoteTarget | None:
    if not url or len(url) > MAX_URL_LENGTH:
        return None
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    host = parsed.hostname.rstrip(".").lower()
    if not host:
        return None
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    return _RemoteTarget(url=url, host=host, port=port or default_port)


class _PublicOnlyResolver:
    def __init__(self, delegate: Resolver | None = None) -> None:
        self._delegate = delegate or aiohttp.resolver.ThreadedResolver()
        self._resolved: dict[tuple[str, int], frozenset[str]] = {}

    async def resolve(
        self, host: str, port: int = 0, family: int = socket.AF_UNSPEC
    ) -> list[dict[str, str | int]]:
        answers = await self._delegate.resolve(host, port, family)
        addresses = frozenset(
            address
            for answer in answers
            if (address := _public_ip(str(answer["host"]))) is not None
        )
        if not answers or len(addresses) != len(answers):
            raise _UnsafeRemoteAddress(host=host)
        self._resolved[(host.rstrip(".").lower(), port)] = addresses
        return answers

    def addresses_for(self, host: str, port: int) -> frozenset[str]:
        return self._resolved.get((host.rstrip(".").lower(), port), frozenset())

    async def close(self) -> None:
        await self._delegate.close()


class _PeerCapturingResponse(aiohttp.ClientResponse):
    connected_peer: str | None = None

    async def start(self, connection: Connection) -> _PeerCapturingResponse:
        transport = connection.transport
        if transport is not None:
            peer = transport.get_extra_info("peername")
            if isinstance(peer, tuple) and peer:
                self.connected_peer = _public_ip(str(peer[0]))
        await super().start(connection)
        return self


def _peer_address(response: aiohttp.ClientResponse) -> str | None:
    connection = response.connection
    transport = getattr(connection, "transport", None)
    if transport is None:
        if isinstance(response, _PeerCapturingResponse):
            return response.connected_peer
        return None
    peer = transport.get_extra_info("peername")
    if not isinstance(peer, tuple) or not peer:
        return None
    return _public_ip(str(peer[0]))


def _verify_peer(
    response: aiohttp.ClientResponse,
    target: _RemoteTarget,
    resolver: _PublicOnlyResolver,
) -> None:
    peer = _peer_address(response)
    literal = _public_ip(target.host)
    expected = (
        frozenset({literal})
        if literal is not None
        else resolver.addresses_for(target.host, target.port)
    )
    if peer is None or not expected:
        raise RemoteReferenceDenied(reason="connected remote peer is unsafe")
    if peer not in expected:
        raise RemoteReferenceDenied(
            reason="connected remote peer changed after resolution"
        )


class AiohttpRemoteReader:
    async def read(self, url: str, max_bytes: int) -> bytes | None:
        current = _target(url)
        if current is None:
            raise RemoteReferenceDenied(reason="invalid or unsafe remote reference")
        resolver = _PublicOnlyResolver()
        timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=15)
        try:
            connector = aiohttp.TCPConnector(
                resolver=resolver,
                use_dns_cache=False,
                force_close=True,
                limit=8,
                limit_per_host=2,
            )
            async with aiohttp.ClientSession(
                timeout=timeout,
                trust_env=False,
                connector=connector,
                response_class=_PeerCapturingResponse,
            ) as session:
                for redirect_count in range(MAX_REDIRECTS + 1):
                    async with session.get(
                        current.url, allow_redirects=False
                    ) as response:
                        _verify_peer(response, current, resolver)
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if location is None or redirect_count >= MAX_REDIRECTS:
                                raise RemoteReferenceDenied(
                                    reason="remote redirect limit exceeded"
                                )
                            following = _target(urljoin(current.url, location))
                            if following is None:
                                raise RemoteReferenceDenied(
                                    reason="remote redirect target is unsafe"
                                )
                            current = following
                            continue
                        if response.status < 200 or response.status >= 300:
                            return None
                        content_length = response.content_length
                        if content_length is not None and content_length > max_bytes:
                            raise RemoteLimitExceeded(max_bytes=max_bytes)
                        payload = bytearray()
                        async for chunk in response.content.iter_chunked(
                            READ_CHUNK_SIZE
                        ):
                            payload.extend(chunk)
                            if len(payload) > max_bytes:
                                raise RemoteLimitExceeded(max_bytes=max_bytes)
                        return bytes(payload)
        except _UnsafeRemoteAddress as exc:
            raise RemoteReferenceDenied(
                reason="remote host resolved to an unsafe address"
            ) from exc
        except (aiohttp.ClientError, TimeoutError, OSError):
            return None
        finally:
            await resolver.close()
        return None

    async def close(self) -> None:
        return None
