from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Endpoint:
    block_id: str
    port: int


@dataclass(frozen=True, slots=True)
class Connection:
    source: Endpoint
    destination: Endpoint
