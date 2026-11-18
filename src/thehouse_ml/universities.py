"""ML-02 T07: verificación de las coordenadas del seed contra OpenStreetMap.

El seed de TheHouse-Backend (`TheHouse.Application/Common/UniversitySeedData.cs`,
commit `b7d75955513320a0a97aa5daf6b3bbe92e9514d2`) se copia a
``data/seed/universities_seed.csv``. Por cada universidad se hace **una** consulta
a Nominatim con User-Agent propio, viewbox de ±0.1° alrededor de la coordenada
del seed, ``bounded=1`` y como máximo 1 petición por segundo (política de uso).
Las respuestas ``jsonv2`` se guardan en ``data/raw/openstreetmap/<code>.json``
(fuera de git) y su SHA-256 en ``data/manifests/openstreetmap.json``.

``prepare`` elige el elemento educativo (``category``/``class=amenity`` y
``type=university|college|school``) más cercano al seed y escribe
``data/universities.csv`` y ``reports/universities_check.md``. Distancias mayores
a 200 m y ausencia de coincidencia se reportan para corregir NV-11 en el backend.

Atribución: © OpenStreetMap contributors (ODbL).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from thehouse_ml.manifests import Manifest, ManifestFile, load_manifest, manifest_path, write_manifest
from thehouse_ml.paths import ProjectPaths

OSM_SOURCE = "openstreetmap"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "TheHouse-ML/0.1 (proyecto escolar UTTT)"
RATE_LIMIT_SECONDS = 1.0
VIEWBOX_DEGREES = 0.1
RESULT_LIMIT = 5
COUNTRY_CODES = "mx"
DISTANCE_THRESHOLD_M = 200.0
EXPECTED_UNIVERSITIES = 10
EARTH_RADIUS_M = 6_371_000.0
EDUCATIONAL_TYPES = frozenset({"university", "college", "school"})

SEED_COLUMNS = ("code", "name", "city", "latitude", "longitude")
UNIVERSITY_COLUMNS = (
    "code",
    "name",
    "seed_lat",
    "seed_lon",
    "osm_type",
    "osm_id",
    "osm_name",
    "osm_lat",
    "osm_lon",
    "distance_m",
    "status",
    "verified_at",
)

OsmFetcher = Callable[[str, Path], None]
Sleeper = Callable[[float], None]


class OsmError(RuntimeError):
    """Falta una respuesta OSM, el seed o los datos tienen un formato inesperado."""


@dataclass(frozen=True)
class OsmOutcome:
    """Resultado de descargar o verificar las respuestas OSM."""

    source_id: str
    status: str
    message: str

    def render(self) -> str:
        """Línea para la consola."""
        return f"{self.source_id}: {self.status} - {self.message}"


@dataclass(frozen=True)
class UniversitySeed:
    """Una universidad del seed de TheHouse-Backend."""

    code: str
    name: str
    city: str
    latitude: str
    longitude: str

    @property
    def latitude_float(self) -> float:
        """Latitud numérica para el viewbox y la distancia."""
        return _as_float(self.latitude, self.code, "latitude")

    @property
    def longitude_float(self) -> float:
        """Longitud numérica para el viewbox y la distancia."""
        return _as_float(self.longitude, self.code, "longitude")


@dataclass(frozen=True)
class OsmCandidate:
    """Un resultado ``jsonv2`` de Nominatim que puede ser un campus."""

    osm_type: str
    osm_id: str
    osm_name: str
    osm_lat: str
    osm_lon: str
    category: str
    kind: str

    @property
    def is_educational(self) -> bool:
        """Indica si el elemento es una universidad, facultad o escuela."""
        return self.category == "amenity" and self.kind in EDUCATIONAL_TYPES


@dataclass(frozen=True)
class UniversityRow:
    """Una fila de ``data/universities.csv``."""

    code: str
    name: str
    seed_lat: str
    seed_lon: str
    osm_type: str = ""
    osm_id: str = ""
    osm_name: str = ""
    osm_lat: str = ""
    osm_lon: str = ""
    distance_m: str = ""
    status: str = ""
    verified_at: str = ""


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos WGS84 (radio medio 6 371 000 m)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lon2 - lon1)
    value = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    value = min(1.0, max(0.0, value))
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def load_seed(path: Path) -> tuple[UniversitySeed, ...]:
    """Lee el seed versionado; valida las cinco columnas y descarta filas vacías."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = tuple(reader.fieldnames or ())
            missing = set(SEED_COLUMNS) - set(header)
            if missing:
                raise OsmError(f"{path.name}: faltan columnas {', '.join(sorted(missing))}")
            seeds = tuple(
                UniversitySeed(
                    code=(row.get("code") or "").strip(),
                    name=(row.get("name") or "").strip(),
                    city=(row.get("city") or "").strip(),
                    latitude=(row.get("latitude") or "").strip(),
                    longitude=(row.get("longitude") or "").strip(),
                )
                for row in reader
                if (row.get("code") or "").strip()
            )
    except OSError as exc:
        raise OsmError(f"No se puede leer el seed {path}: {exc}") from exc
    if not seeds:
        raise OsmError(f"{path.name}: el seed no tiene universidades")
    return seeds


def build_query_url(seed: UniversitySeed) -> str:
    """Construye la única consulta permitida por universidad (Nominatim jsonv2)."""
    west = seed.longitude_float - VIEWBOX_DEGREES
    east = seed.longitude_float + VIEWBOX_DEGREES
    north = seed.latitude_float + VIEWBOX_DEGREES
    south = seed.latitude_float - VIEWBOX_DEGREES
    params = {
        "q": seed.name,
        "format": "jsonv2",
        "limit": RESULT_LIMIT,
        "countrycodes": COUNTRY_CODES,
        "viewbox": f"{west:.6f},{north:.6f},{east:.6f},{south:.6f}",
        "bounded": 1,
    }
    return f"{NOMINATIM_URL}?{urllib.parse.urlencode(params)}"


def fetch_osm(url: str, destination: Path) -> None:
    """Descarga una respuesta jsonv2 con el User-Agent obligatorio."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 - URL construida por build_query_url
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - URL construida por build_query_url
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OsmError(f"No se pudo consultar Nominatim: {exc}") from exc
        partial.write_bytes(payload)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def download_osm(
    source_id: str,
    paths: ProjectPaths,
    fetcher: OsmFetcher | None = None,
    today: str | None = None,
    sleeper: Sleeper | None = None,
) -> OsmOutcome:
    """Descarga las 10 respuestas que falten y escribe su manifiesto SHA-256."""
    seeds = load_seed(paths.data_seed)
    previous = load_manifest(manifest_path(paths, source_id))
    previous_files = {file.filename: file for file in previous.files} if previous else {}
    download_date = today or date.today().isoformat()
    fetch = fetcher or fetch_osm
    sleep = sleeper or time.sleep
    entries: list[ManifestFile] = []
    downloaded = 0
    cached = 0
    requests = 0

    for seed in seeds:
        destination = paths.data_raw / source_id / f"{seed.code}.json"
        if destination.is_file():
            cached += 1
        else:
            if requests:
                sleep(RATE_LIMIT_SECONDS)
            fetch(build_query_url(seed), destination)
            requests += 1
            downloaded += 1
        digest, size = _digest_and_size(destination)
        _check_previous(destination, previous_files.get(destination.name), digest)
        entries.append(
            ManifestFile(
                url=build_query_url(seed),
                filename=destination.name,
                downloaded_at=download_date,
                size_bytes=size,
                sha256=digest,
                license="Open Database License (ODbL) 1.0",
            )
        )

    write_manifest(manifest_path(paths, source_id), Manifest(source_id, tuple(entries)))
    if downloaded and cached:
        status, message = "mixto", f"{len(entries)} respuesta(s) verificadas"
    elif downloaded:
        status, message = "descargado", f"{len(entries)} respuesta(s) verificadas"
    else:
        status, message = "manual", f"{len(entries)} respuesta(s) verificadas"
    return OsmOutcome(source_id, status, message)


def parse_candidates(payload: object, code: str) -> tuple[OsmCandidate, ...]:
    """Convierte una respuesta jsonv2 en candidatos; ignora entradas mal formadas."""
    if not isinstance(payload, list):
        raise OsmError(f"{code}.json: se esperaba una lista de resultados")
    candidates: list[OsmCandidate] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        candidates.append(
            OsmCandidate(
                osm_type=str(item.get("osm_type", "")),
                osm_id=str(item.get("osm_id", "")),
                osm_name=str(item.get("name") or ""),
                osm_lat=str(item.get("lat", "")),
                osm_lon=str(item.get("lon", "")),
                # jsonv2 publica `category` en el servicio actual y `class` en versiones anteriores.
                category=str(item.get("category") or item.get("class") or ""),
                kind=str(item.get("type") or ""),
            )
        )
    return tuple(candidates)


def select_candidate(candidates: Sequence[OsmCandidate], seed: UniversitySeed) -> OsmCandidate | None:
    """Devuelve el candidato educativo más cercano al seed, o ``None``."""
    educational = [candidate for candidate in candidates if candidate.is_educational]
    if not educational:
        return None
    return min(educational, key=lambda candidate: _candidate_distance(candidate, seed))


def build_rows(
    seeds: Sequence[UniversitySeed],
    responses: Mapping[str, Sequence[OsmCandidate]],
    verified_at: str,
) -> tuple[UniversityRow, ...]:
    """Genera las filas con distancia, estado y fecha de verificación."""
    rows: list[UniversityRow] = []
    for seed in seeds:
        chosen = select_candidate(responses.get(seed.code, ()), seed)
        if chosen is None:
            rows.append(
                UniversityRow(
                    code=seed.code,
                    name=seed.name,
                    seed_lat=seed.latitude,
                    seed_lon=seed.longitude,
                    status="sin_coincidencia",
                    verified_at=verified_at,
                )
            )
            continue
        distance = haversine_m(
            seed.latitude_float,
            seed.longitude_float,
            _candidate_float(chosen.osm_lat),
            _candidate_float(chosen.osm_lon),
        )
        status = "ok" if distance <= DISTANCE_THRESHOLD_M else "revisar"
        rows.append(
            UniversityRow(
                code=seed.code,
                name=seed.name,
                seed_lat=seed.latitude,
                seed_lon=seed.longitude,
                osm_type=chosen.osm_type,
                osm_id=chosen.osm_id,
                osm_name=chosen.osm_name,
                osm_lat=chosen.osm_lat,
                osm_lon=chosen.osm_lon,
                distance_m=f"{distance:.2f}",
                status=status,
                verified_at=verified_at,
            )
        )
    return tuple(rows)


def write_universities_csv(path: Path, rows: Sequence[UniversityRow]) -> None:
    """Escribe el CSV versionado con columnas estables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(UNIVERSITY_COLUMNS))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def render_report(rows: Sequence[UniversityRow]) -> str:
    """Reporte con la tabla de las 10 universidades y los casos a corregir."""
    lines = [
        "# Verificación de coordenadas de universidades — ML-02 T07",
        "",
        "Generado por `uv run python -m thehouse_ml prepare` a partir del seed de TheHouse-Backend",
        "(`TheHouse.Application/Common/UniversitySeedData.cs`, commit `b7d75955513320a0a97aa5daf6b3bbe92e9514d2`)",
        "y de las respuestas de Nominatim (OpenStreetMap), una consulta por universidad.",
        "",
        "**Atribución: © OpenStreetMap contributors (ODbL).**",
        "",
        "Distancia haversine entre la coordenada del seed y el elemento educativo de OSM más cercano.",
        f"`ok` si la distancia es ≤ {DISTANCE_THRESHOLD_M:.0f} m; `revisar` si es mayor; `sin_coincidencia`",
        "si no hay un elemento OSM con `category/class=amenity` y `type=university|college|school`.",
        "Las coincidencias dudosas se confirman en openstreetmap.org antes de usarlas en la T05.",
        "",
        "| Código | Universidad | Seed (lat, lon) | OSM (lat, lon) | Distancia (m) | Estado |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        seed_coords = f"{row.seed_lat}, {row.seed_lon}"
        osm_coords = f"{row.osm_lat}, {row.osm_lon}" if row.osm_lat and row.osm_lon else "—"
        distance = row.distance_m or "—"
        lines.append(f"| `{row.code}` | {row.name} | {seed_coords} | {osm_coords} | {distance} | {row.status} |")

    issues = [row for row in rows if row.status in {"revisar", "sin_coincidencia"}]
    lines.extend(["", "## Universidades a corregir en NV-11", ""])
    if issues:
        for row in issues:
            detail = f"{row.distance_m} m" if row.distance_m else "sin candidato educativo"
            lines.append(f"- `{row.code}` — {row.name}: **{row.status}** ({detail}).")
    else:
        lines.append("- Ninguna: todas las distancias están dentro del umbral.")
    lines.append("")
    return "\n".join(lines)


def run_universities(paths: ProjectPaths, today: str | None = None) -> str:
    """Etapa T07: valida las 10 respuestas y escribe el CSV y el reporte."""
    seeds = load_seed(paths.data_seed)
    if len(seeds) != EXPECTED_UNIVERSITIES:
        raise OsmError(f"El seed debe tener {EXPECTED_UNIVERSITIES} universidades; se encontraron {len(seeds)}")
    responses: dict[str, tuple[OsmCandidate, ...]] = {}
    for seed in seeds:
        payload_path = paths.data_raw / OSM_SOURCE / f"{seed.code}.json"
        if not payload_path.is_file():
            raise OsmError(f"falta {payload_path.as_posix()}; ejecute primero `download --source openstreetmap`")
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OsmError(f"No se puede leer {payload_path.name}: {exc}") from exc
        responses[seed.code] = parse_candidates(payload, seed.code)

    rows = build_rows(seeds, responses, today or date.today().isoformat())
    write_universities_csv(paths.universities_csv, rows)
    report_path = paths.reports / "universities_check.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(rows), encoding="utf-8")
    return f"universities: {paths.universities_csv.as_posix()} ({len(rows)} filas) - {report_path.as_posix()}"


def osm_headers(paths: ProjectPaths) -> tuple[set[str], bool]:
    """Claves presentes en las respuestas jsonv2 y si hay respuestas descargadas."""
    folder = paths.data_raw / OSM_SOURCE
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    headers: set[str] = set()
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    headers.update(str(key) for key in item)
    return headers, bool(files)


def _as_float(value: str, code: str, field: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise OsmError(f"{code}: {field} no es numérico ({value!r})") from exc


def _candidate_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.inf


def _candidate_distance(candidate: OsmCandidate, seed: UniversitySeed) -> float:
    return haversine_m(
        seed.latitude_float,
        seed.longitude_float,
        _candidate_float(candidate.osm_lat),
        _candidate_float(candidate.osm_lon),
    )


def _digest_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _check_previous(destination: Path, previous: ManifestFile | None, digest: str) -> None:
    if previous is not None and previous.sha256 != digest:
        raise OsmError(
            f"{destination.name}: el manifiesto existente registra SHA-256 {previous.sha256}, obtenido {digest}"
        )


__all__ = [
    "COUNTRY_CODES",
    "DISTANCE_THRESHOLD_M",
    "EDUCATIONAL_TYPES",
    "EXPECTED_UNIVERSITIES",
    "NOMINATIM_URL",
    "OSM_SOURCE",
    "RATE_LIMIT_SECONDS",
    "RESULT_LIMIT",
    "UNIVERSITY_COLUMNS",
    "USER_AGENT",
    "VIEWBOX_DEGREES",
    "OsmCandidate",
    "OsmError",
    "OsmOutcome",
    "UniversityRow",
    "UniversitySeed",
    "build_query_url",
    "build_rows",
    "download_osm",
    "fetch_osm",
    "haversine_m",
    "load_seed",
    "osm_headers",
    "parse_candidates",
    "render_report",
    "run_universities",
    "select_candidate",
    "write_universities_csv",
]
