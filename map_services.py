"""Geocoding and road-distance helpers backed by OpenStreetMap services."""

from __future__ import annotations

import time
from threading import Lock
from typing import Iterable, Optional

import requests


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_TABLE_URL = "http://router.project-osrm.org/table/v1/driving"
USER_AGENT = "RestaurantRecommendationStudentProject/1.0"

_nominatim_lock = Lock()
_last_nominatim_request = 0.0


def fetch_json(url: str, params: Optional[dict[str, str]] = None) -> object:
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def geocode_address(address: str) -> Optional[dict[str, object]]:
    """Geocode one address while respecting the public Nominatim rate limit."""

    global _last_nominatim_request

    query = address.strip()
    if not query:
        return None
    if "bangalore" not in query.lower() and "bengaluru" not in query.lower():
        query = f"{query}, Bengaluru, Karnataka, India"

    with _nominatim_lock:
        wait_seconds = max(0.0, 1.05 - (time.monotonic() - _last_nominatim_request))
        if wait_seconds:
            time.sleep(wait_seconds)
        result = fetch_json(
            NOMINATIM_URL,
            {
                "q": query,
                "format": "jsonv2",
                "limit": "1",
                "countrycodes": "in",
                "addressdetails": "1",
            },
        )
        _last_nominatim_request = time.monotonic()

    if not isinstance(result, list) or not result:
        return None
    place = result[0]
    return {
        "latitude": float(place["lat"]),
        "longitude": float(place["lon"]),
        "display_name": place.get("display_name", query),
    }


def road_distances(
    origin: tuple[float, float],
    destinations: Iterable[tuple[float, float]],
) -> list[Optional[dict[str, float]]]:
    """Return OSRM driving distance and duration from one origin."""

    destination_list = list(destinations)
    if not destination_list:
        return []

    coordinates = [origin] + destination_list
    coordinate_string = ";".join(f"{lon},{lat}" for lat, lon in coordinates)
    result = fetch_json(
        f"{OSRM_TABLE_URL}/{coordinate_string}",
        {
            "sources": "0",
            "destinations": ";".join(str(index) for index in range(1, len(coordinates))),
            "annotations": "distance,duration",
        },
    )
    if not isinstance(result, dict) or result.get("code") != "Ok":
        return [None] * len(destination_list)

    distances = result.get("distances", [[]])[0]
    durations = result.get("durations", [[]])[0]
    routes = []
    for distance, duration in zip(distances, durations):
        if distance is None or duration is None:
            routes.append(None)
        else:
            routes.append(
                {
                    "distance_km": float(distance) / 1000,
                    "duration_min": float(duration) / 60,
                }
            )
    return routes
