"""Hybrid multi-criteria restaurant recommendation for Zomato Bangalore data.

The model combines:
- constraint-based filtering for budget, distance and cuisine
- Haversine distance from the user's area to each restaurant area
- TF-IDF content similarity over cuisines, restaurant type, location and reviews
- Bayesian rating and weighted multi-criteria ranking
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
except ModuleNotFoundError:  # pragma: no cover - exercised when sklearn is unavailable
    TfidfVectorizer = None
    linear_kernel = None


AREA_COORDINATES = {
    "BTM": (12.9166, 77.6101),
    "Banashankari": (12.9255, 77.5468),
    "Bannerghatta Road": (12.8933, 77.5989),
    "Banaswadi": (13.0142, 77.6519),
    "Basavanagudi": (12.9420, 77.5754),
    "Bellandur": (12.9352, 77.6669),
    "Bommanahalli": (12.8997, 77.6269),
    "Brigade Road": (12.9700, 77.6067),
    "Brookefield": (12.9663, 77.7169),
    "Church Street": (12.9751, 77.6047),
    "Commercial Street": (12.9822, 77.6083),
    "Cunningham Road": (12.9870, 77.5948),
    "Domlur": (12.9611, 77.6387),
    "Ejipura": (12.9452, 77.6269),
    "Electronic City": (12.8452, 77.6602),
    "Frazer Town": (12.9989, 77.6143),
    "HSR": (12.9116, 77.6389),
    "Indiranagar": (12.9784, 77.6408),
    "Infantry Road": (12.9820, 77.6006),
    "Jayanagar": (12.9250, 77.5938),
    "Jeevan Bhima Nagar": (12.9662, 77.6577),
    "JP Nagar": (12.9063, 77.5857),
    "Kalyan Nagar": (13.0221, 77.6408),
    "Kammanahalli": (13.0159, 77.6379),
    "Koramangala": (12.9352, 77.6245),
    "Koramangala 1st Block": (12.9279, 77.6271),
    "Koramangala 3rd Block": (12.9274, 77.6264),
    "Koramangala 4th Block": (12.9346, 77.6305),
    "Koramangala 5th Block": (12.9349, 77.6229),
    "Koramangala 6th Block": (12.9386, 77.6228),
    "Koramangala 7th Block": (12.9363, 77.6155),
    "Koramangala 8th Block": (12.9416, 77.6176),
    "Kumaraswamy Layout": (12.9081, 77.5553),
    "Lavelle Road": (12.9719, 77.5994),
    "Malleshwaram": (13.0031, 77.5643),
    "Marathahalli": (12.9591, 77.6974),
    "MG Road": (12.9755, 77.6068),
    "Nagawara": (13.0435, 77.6209),
    "New BEL Road": (13.0291, 77.5704),
    "Old Airport Road": (12.9580, 77.6583),
    "Rajajinagar": (12.9915, 77.5550),
    "Race Course Road": (12.9844, 77.5857),
    "Residency Road": (12.9716, 77.6033),
    "Richmond Road": (12.9662, 77.6052),
    "Sarjapur Road": (12.9141, 77.6838),
    "Seshadripuram": (12.9935, 77.5787),
    "Shanti Nagar": (12.9576, 77.5977),
    "Shivajinagar": (12.9857, 77.6057),
    "St. Marks Road": (12.9716, 77.6012),
    "Thippasandra": (12.9739, 77.6508),
    "Ulsoor": (12.9817, 77.6284),
    "Vasanth Nagar": (12.9911, 77.5920),
    "Whitefield": (12.9698, 77.7500),
    "Wilson Garden": (12.9487, 77.5968),
    "Yeshwantpur": (13.0250, 77.5340),
}


DEFAULT_WEIGHTS = {
    "rating": 0.35,
    "distance": 0.25,
    "price": 0.20,
    "popularity": 0.10,
    "similarity": 0.10,
}


@dataclass(frozen=True)
class UserPreference:
    """Input constraints and preferences for one recommendation request."""

    budget: float
    max_distance_km: float
    cuisine: str
    current_area: Optional[str] = None
    current_latitude: Optional[float] = None
    current_longitude: Optional[float] = None
    top_n: int = 10
    budget_is_per_person: bool = False


def clean_cost(series: pd.Series) -> pd.Series:
    """Convert Zomato cost strings such as '1,200' to 1200.0."""

    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def clean_rating(series: pd.Series) -> pd.Series:
    """Convert ratings such as '4.1/5' to 4.1 and remove NEW/- values."""

    cleaned = series.astype(str).str.replace("/5", "", regex=False).str.strip()
    cleaned = cleaned.replace({"NEW": np.nan, "-": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_zomato_data(path: str | Path = "data/zomato.csv") -> pd.DataFrame:
    """Load and clean the Kaggle Zomato Bangalore dataset."""

    raw = pd.read_csv(path)
    return prepare_restaurants(raw)


def prepare_restaurants(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean data and add engineered features used by the hybrid recommender."""

    data = raw.copy()
    data = data.drop(columns=["url", "dish_liked", "phone"], errors="ignore")
    data = data.rename(
        columns={
            "approx_cost(for two people)": "cost",
            "listed_in(type)": "type",
            "listed_in(city)": "city",
        }
    )
    data = data.drop_duplicates()

    required = ["name", "location", "cuisines", "cost", "rate", "votes"]
    data = data.dropna(subset=[col for col in required if col in data.columns])
    data["cost"] = clean_cost(data["cost"])
    data["rate"] = clean_rating(data["rate"])
    data["votes"] = pd.to_numeric(data["votes"], errors="coerce").fillna(0)
    data = data.dropna(subset=["cost", "rate"])

    text_columns = ["name", "location", "cuisines", "rest_type", "reviews_list"]
    for column in text_columns:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("").astype(str)

    data["name"] = data["name"].str.title()
    data["estimated_price_per_person"] = data["cost"] / 2
    data = add_area_coordinates(data)
    data["bayesian_rating"] = bayesian_rating(data)
    data["combined_features"] = build_combined_features(data)
    return data.reset_index(drop=True)


def add_area_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    """Attach latitude and longitude for known Bangalore areas."""

    result = data.copy()
    coords = result["location"].map(AREA_COORDINATES)
    result["latitude"] = coords.map(lambda value: value[0] if isinstance(value, tuple) else np.nan)
    result["longitude"] = coords.map(lambda value: value[1] if isinstance(value, tuple) else np.nan)
    return result


def haversine_distance_km(
    lat1: float, lon1: float, lat2: Iterable[float], lon2: Iterable[float]
) -> np.ndarray:
    """Calculate great-circle distance between one point and many points."""

    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def bayesian_rating(data: pd.DataFrame, vote_quantile: float = 0.75) -> pd.Series:
    """IMDB-style weighted rating to reduce bias from very few votes."""

    c = data["rate"].mean()
    m = data["votes"].quantile(vote_quantile)
    v = data["votes"]
    r = data["rate"]
    return ((v / (v + m)) * r + (m / (v + m)) * c).fillna(c)


def build_combined_features(data: pd.DataFrame) -> pd.Series:
    """Combine metadata and reviews for content-based matching."""

    return (
        data["cuisines"].fillna("")
        + " "
        + data["rest_type"].fillna("")
        + " "
        + data["location"].fillna("")
        + " "
        + data["reviews_list"].fillna("")
    ).str.lower()


def resolve_user_coordinates(preference: UserPreference) -> tuple[float, float]:
    """Resolve coordinates from either explicit lat/lon or a known current area."""

    if preference.current_latitude is not None and preference.current_longitude is not None:
        return preference.current_latitude, preference.current_longitude
    if preference.current_area in AREA_COORDINATES:
        return AREA_COORDINATES[preference.current_area]
    known_areas = ", ".join(sorted(AREA_COORDINATES)[:8])
    raise ValueError(
        "Provide current_latitude/current_longitude or a known current_area. "
        f"Examples: {known_areas}, ..."
    )


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize with a stable fallback for constant series."""

    if series.empty:
        return series
    if series.nunique(dropna=True) <= 1:
        return pd.Series(np.ones(len(series)), index=series.index)
    min_value = series.min()
    max_value = series.max()
    values = (series - min_value) / (max_value - min_value)
    return pd.Series(values, index=series.index)


def content_similarity(documents: pd.Series, query: str) -> np.ndarray:
    """Calculate TF-IDF cosine similarity, with a small fallback if sklearn is absent."""

    if TfidfVectorizer is not None and linear_kernel is not None:
        tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), stop_words="english")
        tfidf_matrix = tfidf.fit_transform(pd.concat([documents, pd.Series([query])], ignore_index=True))
        return linear_kernel(tfidf_matrix[:-1], tfidf_matrix[-1]).ravel()

    return fallback_tfidf_similarity(documents.tolist(), query)


def fallback_tfidf_similarity(documents: list[str], query: str) -> np.ndarray:
    """Minimal unigram/bigram TF-IDF cosine similarity used when sklearn is unavailable."""

    all_documents = documents + [query]
    tokenized = [make_ngrams(text) for text in all_documents]
    vocabulary = sorted({token for document in tokenized for token in document})
    if not vocabulary:
        return np.zeros(len(documents))

    token_to_idx = {token: idx for idx, token in enumerate(vocabulary)}
    term_frequency = np.zeros((len(all_documents), len(vocabulary)))
    for row_idx, document in enumerate(tokenized):
        for token in document:
            term_frequency[row_idx, token_to_idx[token]] += 1

    document_frequency = (term_frequency > 0).sum(axis=0)
    idf = np.log((1 + len(all_documents)) / (1 + document_frequency)) + 1
    tfidf = term_frequency * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf = np.divide(tfidf, norms, out=np.zeros_like(tfidf), where=norms != 0)
    return tfidf[:-1] @ tfidf[-1]


def make_ngrams(text: str) -> list[str]:
    """Create simple lowercase word unigrams and bigrams."""

    import re

    words = re.findall(r"[a-z0-9]+", text.lower())
    bigrams = [f"{left} {right}" for left, right in zip(words, words[1:])]
    return words + bigrams


def recommend_restaurants(
    restaurants: pd.DataFrame,
    preference: UserPreference,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Recommend restaurants using constraints, content similarity and ranking."""

    weights = weights or DEFAULT_WEIGHTS
    data = restaurants.copy()
    budget_for_two = preference.budget * 2 if preference.budget_is_per_person else preference.budget
    user_lat, user_lon = resolve_user_coordinates(preference)

    if "distance_km" not in data.columns:
        data["distance_km"] = haversine_distance_km(
            user_lat,
            user_lon,
            data["latitude"].fillna(0),
            data["longitude"].fillna(0),
        )
        data.loc[data[["latitude", "longitude"]].isna().any(axis=1), "distance_km"] = np.inf

    cuisine_pattern = re_escape_query(preference.cuisine)
    filtered = data[
        (data["cost"] <= budget_for_two)
        & (data["distance_km"] <= preference.max_distance_km)
        & (data["cuisines"].str.contains(cuisine_pattern, case=False, na=False, regex=True))
    ].copy()

    if filtered.empty:
        return filtered

    documents = filtered["combined_features"].fillna("")
    query = f"{preference.cuisine} {preference.current_area or ''}".strip().lower()
    tfidf_similarity = content_similarity(documents, query)
    selected_cuisines = [
        cuisine.strip().lower()
        for cuisine in preference.cuisine.split(",")
        if cuisine.strip()
    ]
    filtered["cuisine_match_count"] = filtered["cuisines"].apply(
        lambda value: sum(cuisine in value.lower() for cuisine in selected_cuisines)
    )
    filtered["cuisine_match_ratio"] = (
        filtered["cuisine_match_count"] / max(1, len(selected_cuisines))
    )
    filtered["content_similarity_score"] = (
        0.65 * filtered["cuisine_match_ratio"] + 0.35 * tfidf_similarity
    )

    filtered["rating_score"] = (filtered["bayesian_rating"] / 5).clip(0, 1)
    filtered["distance_score"] = (1 - filtered["distance_km"] / preference.max_distance_km).clip(0, 1)
    filtered["price_score"] = (1 - filtered["cost"] / budget_for_two).clip(0, 1)
    filtered["popularity_score"] = normalize(filtered["votes"])

    filtered["final_score"] = (
        weights["rating"] * filtered["rating_score"]
        + weights["distance"] * filtered["distance_score"]
        + weights["price"] * filtered["price_score"]
        + weights["popularity"] * filtered["popularity_score"]
        + weights["similarity"] * filtered["content_similarity_score"]
    )

    columns = [
        "name",
        "address",
        "location",
        "cuisines",
        "cost",
        "estimated_price_per_person",
        "distance_km",
        "distance_source",
        "latitude",
        "longitude",
        "rate",
        "votes",
        "bayesian_rating",
        "cuisine_match_count",
        "cuisine_match_ratio",
        "content_similarity_score",
        "final_score",
    ]
    columns = [column for column in columns if column in filtered.columns]
    return (
        filtered.sort_values(
            ["cuisine_match_count", "final_score", "bayesian_rating", "votes"],
            ascending=False,
        )
        .drop_duplicates(subset=["name", "location", "cuisines"])
        .head(preference.top_n)[columns]
        .reset_index(drop=True)
    )


def re_escape_query(query: str) -> str:
    """Escape user text while allowing comma-separated cuisine alternatives."""

    import re

    parts = [part.strip() for part in query.split(",") if part.strip()]
    return "|".join(re.escape(part) for part in parts) if parts else ".*"


def evaluate_rule_checking(
    recommendations: pd.DataFrame,
    preference: UserPreference,
) -> dict[str, float]:
    """Check whether top recommendations satisfy hard constraints."""

    if recommendations.empty:
        return {
            "budget_pass_rate": 0.0,
            "distance_pass_rate": 0.0,
            "cuisine_pass_rate": 0.0,
            "precision_at_k": 0.0,
        }

    budget_for_two = preference.budget * 2 if preference.budget_is_per_person else preference.budget
    cuisine_pattern = re_escape_query(preference.cuisine)
    budget_ok = recommendations["cost"] <= budget_for_two
    distance_ok = recommendations["distance_km"] <= preference.max_distance_km
    cuisine_ok = recommendations["cuisines"].str.contains(
        cuisine_pattern, case=False, na=False, regex=True
    )
    relevant = budget_ok & distance_ok & cuisine_ok & (recommendations["rate"] >= 4.0)
    return {
        "budget_pass_rate": float(budget_ok.mean()),
        "distance_pass_rate": float(distance_ok.mean()),
        "cuisine_pass_rate": float(cuisine_ok.mean()),
        "precision_at_k": float(relevant.mean()),
    }


if __name__ == "__main__":
    data_path = Path("data/zomato.csv")
    if not data_path.exists():
        raise SystemExit("Put the Kaggle dataset at data/zomato.csv, then run this script again.")

    restaurants_df = load_zomato_data(data_path)
    prefs = UserPreference(
        budget=500,
        max_distance_km=3,
        cuisine="Biryani",
        current_area="Koramangala",
        top_n=10,
    )
    top = recommend_restaurants(restaurants_df, prefs)
    print(top)
    print(evaluate_rule_checking(top, prefs))
