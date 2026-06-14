from __future__ import annotations

from html import escape
from math import cos, pi, radians, sin
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from hybrid_recommender import (
    AREA_COORDINATES,
    UserPreference,
    haversine_distance_km,
    load_zomato_data,
    re_escape_query,
    recommend_restaurants,
)
from map_services import geocode_address, road_distances


DATA_PATH = Path("data/zomato.csv")

st.set_page_config(page_title="Restaurant Finder", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f6f7f8; }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { color: #17191c; letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: transparent;
        border-left: 3px solid #d94f4f;
        padding-left: 12px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #e2e5e9;
        border-radius: 6px;
    }
    .restaurant-rank {
        color: #d94f4f;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .restaurant-name {
        color: #17191c;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 3px 0 8px;
        overflow-wrap: anywhere;
    }
    .restaurant-meta { color: #62676f; font-size: 0.9rem; overflow-wrap: anywhere; }
    .route-note { color: #727780; font-size: 0.78rem; }
    .restaurant-card {
        background: #ffffff;
        border: 1px solid #e2e5e9;
        border-radius: 6px;
        padding: 18px;
        height: 365px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
    }
    .popular-card { height: 285px; }
    .card-address {
        color: #727780;
        font-size: 0.82rem;
        line-height: 1.4;
        margin-top: 10px;
        overflow-wrap: anywhere;
    }
    .card-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: auto;
        padding-top: 18px;
    }
    .popular-card .card-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card-stat { border-left: 3px solid #d94f4f; padding-left: 9px; }
    .card-stat-label { color: #727780; font-size: 0.72rem; }
    .card-stat-value { color: #25282d; font-size: 1.08rem; font-weight: 700; }
    .card-route { color: #727780; font-size: 0.76rem; margin-top: 14px; }
    .final-score {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #eceef1;
        color: #62676f;
        font-size: 0.8rem;
    }
    .final-score strong { color: #d94f4f; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Preparing restaurant data...")
def load_data(path: str) -> pd.DataFrame:
    return load_zomato_data(path)


@st.cache_data(ttl=60 * 60 * 24 * 30, show_spinner=False)
def cached_geocode(address: str) -> dict[str, object] | None:
    return geocode_address(address)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_routes(
    origin: tuple[float, float],
    destinations: tuple[tuple[float, float], ...],
) -> list[dict[str, float] | None]:
    return road_distances(origin, destinations)


def cuisine_options(restaurants: pd.DataFrame) -> list[str]:
    values = (
        restaurants["cuisines"].dropna().astype(str).str.split(",").explode().str.strip()
    )
    options = sorted(value for value in values.unique() if value)
    preferred = ["Biryani", "South Indian", "North Indian", "Chinese", "Pizza", "Korean", "Cafe"]
    return [value for value in preferred if value in options] + [
        value for value in options if value not in preferred
    ]


def build_shortlist(
    restaurants: pd.DataFrame,
    preference: UserPreference,
    candidate_limit: int,
) -> pd.DataFrame:
    budget_for_two = preference.budget * 2 if preference.budget_is_per_person else preference.budget
    area_lat, area_lon = AREA_COORDINATES[preference.current_area]
    candidates = restaurants[
        (restaurants["cost"] <= budget_for_two)
        & restaurants["cuisines"].str.contains(
            re_escape_query(preference.cuisine),
            case=False,
            na=False,
            regex=True,
        )
        & restaurants["latitude"].notna()
        & restaurants["longitude"].notna()
    ].copy()
    candidates["approx_distance_km"] = haversine_distance_km(
        area_lat,
        area_lon,
        candidates["latitude"],
        candidates["longitude"],
    )
    search_radius = max(preference.max_distance_km * 2.5, 8.0)
    candidates = candidates[candidates["approx_distance_km"] <= search_radius]
    selected_cuisines = [
        cuisine.strip().lower()
        for cuisine in preference.cuisine.split(",")
        if cuisine.strip()
    ]
    candidates["cuisine_match_count"] = candidates["cuisines"].apply(
        lambda value: sum(cuisine in value.lower() for cuisine in selected_cuisines)
    )
    return (
        candidates.sort_values(
            ["cuisine_match_count", "bayesian_rating", "votes"],
            ascending=False,
        )
        .drop_duplicates(subset=["name", "address"])
        .head(candidate_limit)
        .copy()
    )


def add_map_distances(
    candidates: pd.DataFrame,
    origin: tuple[float, float],
) -> pd.DataFrame:
    enriched = candidates.copy()
    coordinates = []
    sources = []

    for row in enriched.itertuples():
        result = None
        try:
            result = cached_geocode(f"{row.name}, {row.address}")
        except Exception:
            result = None

        if result:
            coordinates.append((float(result["latitude"]), float(result["longitude"])))
            sources.append("Exact-address route")
        else:
            coordinates.append((float(row.latitude), float(row.longitude)))
            sources.append("Area-based route")

    routes = []
    try:
        routes = cached_routes(origin, tuple(coordinates))
    except Exception:
        routes = [None] * len(coordinates)

    distances = []
    durations = []
    for coordinate, route in zip(coordinates, routes):
        if route:
            distances.append(route["distance_km"])
            durations.append(route["duration_min"])
        else:
            fallback = float(
                haversine_distance_km(
                    origin[0],
                    origin[1],
                    [coordinate[0]],
                    [coordinate[1]],
                )[0]
            )
            distances.append(max(0.1, fallback * 1.25))
            durations.append(np.nan)

    enriched["latitude"] = [coordinate[0] for coordinate in coordinates]
    enriched["longitude"] = [coordinate[1] for coordinate in coordinates]
    enriched["distance_km"] = distances
    enriched["duration_min"] = durations
    enriched["distance_source"] = sources
    return enriched


def find_restaurants(
    restaurants: pd.DataFrame,
    preference: UserPreference,
    current_address: str,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, tuple[float, float], str]:
    area_coordinates = AREA_COORDINATES[preference.current_area]
    origin = area_coordinates
    resolved_address = preference.current_area

    try:
        geocoded_origin = cached_geocode(current_address)
    except Exception:
        geocoded_origin = None
    if geocoded_origin:
        origin = (
            float(geocoded_origin["latitude"]),
            float(geocoded_origin["longitude"]),
        )
        resolved_address = str(geocoded_origin["display_name"])

    candidate_limit = min(16, max(preference.top_n + 4, 10))
    shortlist = build_shortlist(restaurants, preference, candidate_limit)
    if shortlist.empty:
        return shortlist, origin, resolved_address

    routed = add_map_distances(shortlist, origin)
    routed_preference = UserPreference(
        budget=preference.budget,
        max_distance_km=preference.max_distance_km,
        cuisine=preference.cuisine,
        current_latitude=origin[0],
        current_longitude=origin[1],
        top_n=preference.top_n,
        budget_is_per_person=preference.budget_is_per_person,
    )
    results = recommend_restaurants(routed, routed_preference, weights=weights)
    if "duration_min" in routed.columns and not results.empty:
        duration_lookup = routed.set_index(["name", "address"])["duration_min"]
        results["duration_min"] = [
            duration_lookup.get((row["name"], row["address"]), np.nan)
            for _, row in results.iterrows()
        ]
    return results, origin, resolved_address


def render_restaurant_card(row: pd.Series, rank: int, featured: bool = False) -> None:
    label = "Top pick" if featured else f"#{rank}"
    route_text = row.get("distance_source", "Area-based route")
    duration = row.get("duration_min", np.nan)
    if pd.notna(duration):
        route_text = f"{route_text} · about {duration:.0f} min driving"
    address = row.get("address", "")
    final_score_percent = float(row.get("final_score", 0)) * 100
    st.markdown(
        f"""
        <div class="restaurant-card">
          <div class="restaurant-rank">{escape(label)}</div>
          <div class="restaurant-name">{escape(str(row['name']))}</div>
          <div class="restaurant-meta">{escape(str(row['location']))}</div>
          <div class="restaurant-meta">{escape(str(row['cuisines']))}</div>
          <div class="card-address">{escape(str(address))}</div>
          <div class="card-stats">
            <div class="card-stat"><div class="card-stat-label">Rating</div><div class="card-stat-value">{row['rate']:.1f}</div></div>
            <div class="card-stat"><div class="card-stat-label">Distance</div><div class="card-stat-value">{row['distance_km']:.1f} km</div></div>
            <div class="card-stat"><div class="card-stat-label">For two</div><div class="card-stat-value">₹{row['cost']:.0f}</div></div>
          </div>
          <div class="card-route">{escape(route_text)}</div>
          <div class="final-score"><span>Final score</span><strong>{final_score_percent:.1f}%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_popular_card(row: pd.Series) -> None:
    st.markdown(
        f"""
        <div class="restaurant-card popular-card">
          <div class="restaurant-name">{escape(str(row['name']))}</div>
          <div class="restaurant-meta">{escape(str(row['cuisines']))}</div>
          <div class="card-address">{escape(str(row.get('address', '')))}</div>
          <div class="card-stats">
            <div class="card-stat"><div class="card-stat-label">Rating</div><div class="card-stat-value">{row['rate']:.1f}</div></div>
            <div class="card-stat"><div class="card-stat-label">For two</div><div class="card-stat-value">₹{row['cost']:.0f}</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_map_points(
    recommendations: pd.DataFrame,
    origin: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    restaurant_points = recommendations[["name", "latitude", "longitude", "distance_km"]].copy()
    restaurant_points = restaurant_points.reset_index(drop=True)
    restaurant_points["rank"] = restaurant_points.index + 1
    restaurant_points["distance_display"] = restaurant_points["distance_km"].round(1)
    restaurant_points["color"] = [
        [217, 79, 79, 220] if index < 3 else [235, 181, 55, 220]
        for index in range(len(restaurant_points))
    ]
    restaurant_points["radius"] = 90
    restaurant_points["display_latitude"] = restaurant_points["latitude"]
    restaurant_points["display_longitude"] = restaurant_points["longitude"]

    minimum_marker_gap_km = 0.18
    occupied_positions = [origin]
    for row_index in restaurant_points.index:
        original_latitude = float(restaurant_points.loc[row_index, "latitude"])
        original_longitude = float(restaurant_points.loc[row_index, "longitude"])
        candidate = (original_latitude, original_longitude)

        def has_collision(position: tuple[float, float]) -> bool:
            distances = haversine_distance_km(
                position[0],
                position[1],
                [occupied[0] for occupied in occupied_positions],
                [occupied[1] for occupied in occupied_positions],
            )
            return bool((distances < minimum_marker_gap_km).any())

        if has_collision(candidate):
            longitude_scale = max(cos(radians(original_latitude)), 0.2)
            for ring in range(1, 4):
                radius_km = minimum_marker_gap_km * ring
                for slot in range(12):
                    angle = 2 * pi * slot / 12 + row_index * 0.37
                    alternative = (
                        original_latitude + radius_km / 111.0 * sin(angle),
                        original_longitude
                        + radius_km / (111.0 * longitude_scale) * cos(angle),
                    )
                    if not has_collision(alternative):
                        candidate = alternative
                        break
                if not has_collision(candidate):
                    break

        restaurant_points.loc[row_index, "display_latitude"] = candidate[0]
        restaurant_points.loc[row_index, "display_longitude"] = candidate[1]
        occupied_positions.append(candidate)

    user_point = pd.DataFrame(
        [
            {
                "name": "Your location",
                "latitude": origin[0],
                "longitude": origin[1],
                "display_latitude": origin[0],
                "display_longitude": origin[1],
                "distance_km": 0,
                "distance_display": 0.0,
                "color": [30, 118, 92, 230],
                "radius": 130,
                "rank": "You",
            }
        ]
    )
    points = pd.concat([user_point, restaurant_points], ignore_index=True)
    return restaurant_points, points


def render_map(
    recommendations: pd.DataFrame,
    origin: tuple[float, float],
) -> None:
    restaurant_points, points = prepare_map_points(recommendations, origin)
    layer = pdk.Layer(
        "ScatterplotLayer",
        points,
        get_position="[display_longitude, display_latitude]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
    )
    labels = pdk.Layer(
        "TextLayer",
        restaurant_points,
        get_position="[display_longitude, display_latitude]",
        get_text="rank",
        get_size=13,
        get_color=[255, 255, 255, 255],
        get_text_anchor="middle",
        get_alignment_baseline="center",
        pickable=False,
    )
    view_state = pdk.ViewState(
        latitude=float(points["display_latitude"].mean()),
        longitude=float(points["display_longitude"].mean()),
        zoom=12,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=view_state,
            layers=[layer, labels],
            tooltip={"text": "#{rank} {name}\n{distance_display} km"},
        ),
        use_container_width=True,
    )
    st.caption("Map data © OpenStreetMap contributors. Road distance by OSRM.")
    st.markdown(
        '<span style="color:#d94f4f;font-weight:700">● Best matches</span>'
        '&nbsp;&nbsp;&nbsp;'
        '<span style="color:#ebb537;font-weight:700">● Other recommendations</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Markers sharing the same area coordinate are slightly separated on the map "
        "so every recommended restaurant remains visible."
    )


def go_home() -> None:
    st.session_state["page"] = "home"
    for key in [
        "search_results",
        "search_origin",
        "resolved_address",
        "search_preference",
    ]:
        st.session_state.pop(key, None)


def render_home(restaurants: pd.DataFrame, areas: list[str], cuisines: list[str]) -> None:
    with st.sidebar:
        st.title("Restaurant Finder")
        with st.container(border=True):
            current_area = st.selectbox(
                "Area",
                areas,
                index=areas.index("Koramangala"),
                key="home_area",
            )
            current_address = st.text_input(
                "Your address",
                value=f"{current_area}, Bengaluru",
            )
            selected_cuisines = st.multiselect(
                "Cuisines",
                cuisines,
                default=["Biryani"] if "Biryani" in cuisines else cuisines[:1],
            )
            budget_is_per_person = st.toggle("Budget per person", value=False)
            budget = st.number_input(
                "Maximum budget (INR)",
                min_value=1.0,
                value=600.0,
                step=1.0,
            )
            max_distance_km = st.number_input(
                "Maximum driving distance (km)",
                min_value=0.1,
                value=5.0,
                step=0.1,
            )
            submitted = st.button(
                "Find restaurants",
                type="primary",
                width="stretch",
            )

    st.title(f"Popular in {current_area}")
    popular = (
        restaurants[restaurants["location"].str.contains(current_area, case=False, na=False)]
        .sort_values(["bayesian_rating", "votes"], ascending=False)
        .drop_duplicates(subset=["name", "address"])
        .head(12)
    )
    if popular.empty:
        popular = (
            restaurants.sort_values(["bayesian_rating", "votes"], ascending=False)
            .drop_duplicates(subset=["name", "address"])
            .head(12)
        )
    for start in range(0, len(popular), 3):
        columns = st.columns(3)
        for column, (_, row) in zip(columns, popular.iloc[start : start + 3].iterrows()):
            with column:
                render_popular_card(row)

    if not submitted:
        return

    if not selected_cuisines:
        st.warning("Select at least one cuisine.")
        return

    weights = {
        "rating": 0.35,
        "distance": 0.25,
        "price": 0.20,
        "popularity": 0.10,
        "similarity": 0.10,
    }
    preference = UserPreference(
        budget=float(budget),
        max_distance_km=float(max_distance_km),
        cuisine=", ".join(selected_cuisines),
        current_area=current_area,
        top_n=8,
        budget_is_per_person=budget_is_per_person,
    )
    with st.spinner("Checking addresses and road distances..."):
        results, origin, resolved_address = find_restaurants(
            restaurants,
            preference,
            current_address,
            weights,
        )
    st.session_state["search_results"] = results
    st.session_state["search_origin"] = origin
    st.session_state["resolved_address"] = resolved_address
    st.session_state["search_preference"] = preference
    st.session_state["page"] = "results"
    st.rerun()


def render_results() -> None:
    results = st.session_state["search_results"]
    origin = st.session_state["search_origin"]
    resolved_address = st.session_state["resolved_address"]
    active_preference = st.session_state["search_preference"]

    if st.button("← Back", on_click=go_home):
        st.rerun()

    cuisine_title = " + ".join(
        cuisine.strip() for cuisine in active_preference.cuisine.split(",")
    )
    st.title(f"{cuisine_title} near {active_preference.current_area}")
    st.caption(resolved_address)

    if results.empty:
        st.warning("No restaurants matched this budget and driving distance.")
        return

    st.markdown("### Best matches")
    featured_columns = st.columns(min(3, len(results)))
    for index, column in enumerate(featured_columns):
        with column:
            render_restaurant_card(results.iloc[index], index + 1, featured=index == 0)

    list_tab, map_tab, details_tab = st.tabs(["All results", "Map", "Ranking details"])

    with list_tab:
        for start in range(0, len(results), 2):
            columns = st.columns(2)
            for column, (index, row) in zip(columns, results.iloc[start : start + 2].iterrows()):
                with column:
                    render_restaurant_card(row, index + 1)

    with map_tab:
        render_map(results, origin)

    with details_tab:
        column_names = {
            "name": "Restaurant",
            "location": "Area",
            "cost": "Cost for two (INR)",
            "distance_km": "Driving distance (km)",
            "duration_min": "Estimated drive (min)",
            "rate": "Customer rating",
            "votes": "Number of reviews",
            "bayesian_rating": "Confidence-adjusted rating",
            "cuisine_match_count": "Cuisine preferences matched",
            "cuisine_match_ratio": "Cuisine preference coverage",
            "content_similarity_score": "Combined cuisine match score",
            "final_score": "Overall recommendation score",
            "distance_source": "Distance method",
        }
        detail_columns = [column for column in column_names if column in results.columns]
        display_details = results[detail_columns].rename(columns=column_names).copy()
        numeric_columns = display_details.select_dtypes(include=["number"]).columns
        display_details[numeric_columns] = display_details[numeric_columns].round(3)
        st.dataframe(
            display_details,
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    if not DATA_PATH.exists():
        st.error("Missing data/zomato.csv.")
        st.stop()

    restaurants = load_data(str(DATA_PATH))
    areas = sorted(AREA_COORDINATES)
    cuisines = cuisine_options(restaurants)
    st.session_state.setdefault("page", "home")

    if st.session_state["page"] == "results" and "search_results" in st.session_state:
        render_results()
    else:
        render_home(restaurants, areas, cuisines)


if __name__ == "__main__":
    main()
