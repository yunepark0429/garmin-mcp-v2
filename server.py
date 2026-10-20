"""Garmin Connect MCP server - extended analytics build."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import tarfile
import tempfile
import threading
from datetime import date, timedelta
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import StaticTokenVerifier
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)
from starlette.requests import Request
from starlette.responses import PlainTextResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("garmin-mcp")

TOKENSTORE = os.environ.get(
    "GARMINTOKENS", os.path.join(tempfile.gettempdir(), ".garminconnect")
)


def _bootstrap_tokenstore_from_env() -> None:
    blob = os.environ.get("GARMINTOKENS_BASE64")
    if not blob or (os.path.isdir(TOKENSTORE) and os.listdir(TOKENSTORE)):
        return
    os.makedirs(TOKENSTORE, exist_ok=True)
    try:
        raw = base64.b64decode(blob)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            tar.extractall(TOKENSTORE, filter="data")
        log.info("Loaded Garmin tokens into %s", TOKENSTORE)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to extract Garmin tokenstore: %s", exc)


_client: Garmin | None = None
_client_lock = threading.Lock()


def garmin() -> Garmin:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client

        _bootstrap_tokenstore_from_env()
        have_tokens = os.path.isdir(TOKENSTORE) and bool(os.listdir(TOKENSTORE))

        email = os.environ.get("GARMIN_EMAIL")
        password = os.environ.get("GARMIN_PASSWORD")
        if not have_tokens and (not email or not password):
            raise RuntimeError(
                "No Garmin tokenstore found. Set GARMINTOKENS_BASE64, "
                "or set both GARMIN_EMAIL and GARMIN_PASSWORD."
            )

        def _no_mfa() -> str:
            raise RuntimeError(
                "Garmin is requesting MFA but the server is non-interactive."
            )

        client = Garmin(email=email or "", password=password or "", prompt_mfa=_no_mfa)
        try:
            client.login(TOKENSTORE)
        except GarminConnectAuthenticationError as exc:
            raise RuntimeError(f"Garmin authentication failed: {exc}") from exc
        except GarminConnectTooManyRequestsError as exc:
            raise RuntimeError(f"Garmin rate-limited the login: {exc}") from exc
        except GarminConnectConnectionError as exc:
            raise RuntimeError(f"Garmin connection error: {exc}") from exc

        _client = client
        return _client


def _iso(day: str | None) -> str:
    return day or date.today().isoformat()


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    out: list[str] = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


async def _fetch_range(fn, days: list[str], key: str) -> list[dict[str, Any]]:
    results = await asyncio.gather(
        *(asyncio.to_thread(fn, d) for d in days),
        return_exceptions=True,
    )
    out: list[dict[str, Any]] = []
    for d, r in zip(days, results):
        if isinstance(r, Exception):
            out.append({"date": d, "error": str(r)})
        else:
            out.append({"date": d, key: r})
    return out


def _safe_optional(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return {"ok": True, "data": fn(*args, **kwargs), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "data": None, "error": str(exc)}


auth_token = os.environ.get("MCP_BEARER_TOKEN")
if auth_token:
    auth = StaticTokenVerifier(
        tokens={auth_token: {"sub": "garmin-mcp", "client_id": "garmin-mcp"}}
    )
    mcp = FastMCP("Garmin MCP Extended", auth=auth)
    log.info("Bearer auth enabled")
else:
    mcp = FastMCP("Garmin MCP Extended")
    log.warning("MCP_BEARER_TOKEN not set - server is UNAUTHENTICATED")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


# ---------- Existing mirror tools ----------

@mcp.tool
def get_user_profile() -> dict[str, Any]:
    c = garmin()
    return {
        "display_name": c.display_name,
        "full_name": c.full_name,
        "unit_system": c.unit_system,
    }


@mcp.tool
def get_daily_stats(day: str | None = None) -> dict[str, Any]:
    return garmin().get_stats(_iso(day))


@mcp.tool
def get_steps_intraday(day: str | None = None) -> list[dict[str, Any]]:
    return garmin().get_steps_data(_iso(day))


@mcp.tool
def get_daily_steps(start_date: str, end_date: str) -> list[dict[str, Any]]:
    return garmin().get_daily_steps(start_date, end_date)


@mcp.tool
def get_heart_rates(day: str | None = None) -> dict[str, Any]:
    return garmin().get_heart_rates(_iso(day))


@mcp.tool
def get_sleep_data(day: str | None = None) -> dict[str, Any]:
    return garmin().get_sleep_data(_iso(day))


@mcp.tool
def get_stress_data(day: str | None = None) -> dict[str, Any]:
    return garmin().get_stress_data(_iso(day))


@mcp.tool
def get_body_battery(day: str | None = None) -> Any:
    d = _iso(day)
    return garmin().get_body_battery(d, d)


@mcp.tool
def get_body_composition(day: str | None = None) -> dict[str, Any]:
    return garmin().get_body_composition(_iso(day))


@mcp.tool
def get_floors(day: str | None = None) -> dict[str, Any]:
    return garmin().get_floors(_iso(day))


@mcp.tool
def get_hrv_data(day: str | None = None) -> dict[str, Any]:
    return garmin().get_hrv_data(_iso(day))


@mcp.tool
def get_training_readiness(day: str | None = None) -> list[dict[str, Any]]:
    return garmin().get_training_readiness(_iso(day))


@mcp.tool
def get_recent_activities(limit: int = 10) -> list[dict[str, Any]]:
    return garmin().get_activities(0, limit)


@mcp.tool
def get_activities_by_date(
    start_date: str,
    end_date: str,
    activity_type: str | None = None,
) -> list[dict[str, Any]]:
    return garmin().get_activities_by_date(start_date, end_date, activity_type)


@mcp.tool
async def get_sleep_data_range(start_date: str, end_date: str) -> list[dict[str, Any]]:
    return await _fetch_range(
        garmin().get_sleep_data, _date_range(start_date, end_date), "sleep"
    )


@mcp.tool
async def get_hrv_data_range(start_date: str, end_date: str) -> list[dict[str, Any]]:
    return await _fetch_range(
        garmin().get_hrv_data, _date_range(start_date, end_date), "hrv"
    )


@mcp.tool
async def get_stress_data_range(start_date: str, end_date: str) -> list[dict[str, Any]]:
    return await _fetch_range(
        garmin().get_stress_data, _date_range(start_date, end_date), "stress"
    )


@mcp.tool
async def get_heart_rates_range(start_date: str, end_date: str) -> list[dict[str, Any]]:
    return await _fetch_range(
        garmin().get_heart_rates, _date_range(start_date, end_date), "heart_rates"
    )


@mcp.tool
async def get_training_readiness_range(
    start_date: str, end_date: str
) -> list[dict[str, Any]]:
    return await _fetch_range(
        garmin().get_training_readiness,
        _date_range(start_date, end_date),
        "training_readiness",
    )


@mcp.tool
def get_body_battery_range(start_date: str, end_date: str) -> Any:
    return garmin().get_body_battery(start_date, end_date)


@mcp.tool
async def get_last_n_days_summary(days: int = 7) -> list[dict[str, Any]]:
    c = garmin()
    today = date.today()
    iso_days = [(today - timedelta(days=i)).isoformat() for i in range(days)]
    return await _fetch_range(c.get_stats, iso_days, "stats")


# ---------- Detailed activity analytics ----------

@mcp.tool
def get_activity_summary(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity(activity_id)


@mcp.tool
def get_activity_details(
    activity_id: str,
    maxchart: int = 10000,
    maxpoly: int = 0,
) -> dict[str, Any]:
    """Detailed time-series chart data. maxpoly=0 avoids unnecessary route geometry."""
    return garmin().get_activity_details(activity_id, maxchart=maxchart, maxpoly=maxpoly)


@mcp.tool
def get_activity_laps(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity_laps(activity_id)


@mcp.tool
def get_activity_splits(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity_splits(activity_id)


@mcp.tool
def get_activity_split_summaries(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity_split_summaries(activity_id)


@mcp.tool
def get_activity_hr_zones(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity_hr_in_timezones(activity_id)


@mcp.tool
def get_activity_power_zones(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity_power_in_timezones(activity_id)


@mcp.tool
def get_activity_power_curve(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity_power_curve(activity_id)


@mcp.tool
def get_activity_segments(activity_id: str) -> list[dict[str, Any]]:
    return garmin().get_activity_segments(activity_id)


@mcp.tool
def get_activity_weather(activity_id: str) -> dict[str, Any]:
    return garmin().get_activity_weather(activity_id)


@mcp.tool
def get_activity_subsecond_details(
    activity_id: str,
    maxchart: int = 10000,
) -> dict[str, Any]:
    return garmin().get_activity_subsecond_details(activity_id, maxchart=maxchart)


@mcp.tool
def get_full_activity_analysis_bundle(
    activity_id: str,
    maxchart: int = 10000,
) -> dict[str, Any]:
    """One-call analysis bundle. Optional subresources return per-part errors instead of failing the whole bundle."""
    c = garmin()
    return {
        "activity_id": str(activity_id),
        "summary": _safe_optional(c.get_activity, activity_id),
        "details": _safe_optional(c.get_activity_details, activity_id, maxchart=maxchart, maxpoly=0),
        "laps": _safe_optional(c.get_activity_laps, activity_id),
        "splits": _safe_optional(c.get_activity_splits, activity_id),
        "split_summaries": _safe_optional(c.get_activity_split_summaries, activity_id),
        "hr_zones": _safe_optional(c.get_activity_hr_in_timezones, activity_id),
        "power_zones": _safe_optional(c.get_activity_power_in_timezones, activity_id),
        "power_curve": _safe_optional(c.get_activity_power_curve, activity_id),
        "segments": _safe_optional(c.get_activity_segments, activity_id),
        "weather": _safe_optional(c.get_activity_weather, activity_id),
    }


@mcp.tool
def download_activity_original_b64(activity_id: str) -> dict[str, Any]:
    """Return Garmin's original activity download as base64 so a client can archive the exact file."""
    c = garmin()
    raw = c.download_activity(activity_id, c.ActivityDownloadFormat.ORIGINAL)
    return {
        "activity_id": str(activity_id),
        "encoding": "base64",
        "content_b64": base64.b64encode(raw).decode("ascii"),
        "byte_length": len(raw),
    }


# ---------- Training / cycling analytics ----------

@mcp.tool
def get_cycling_ftp() -> Any:
    return garmin().get_cycling_ftp()


@mcp.tool
def get_ftp_history(
    start_date: str,
    end_date: str,
    aggregation: str = "daily",
) -> Any:
    return garmin().get_functional_threshold_power_range(
        start_date, end_date, sport="CYCLING", aggregation=aggregation
    )


@mcp.tool
def get_heart_rate_zones() -> Any:
    return garmin().get_heart_rate_zones()


@mcp.tool
def get_power_zones() -> Any:
    return garmin().get_power_zones()


@mcp.tool
def get_cycling_power_zones() -> Any:
    return garmin().get_power_zones_for_sport("CYCLING")


@mcp.tool
def get_training_status(day: str | None = None) -> dict[str, Any]:
    return garmin().get_training_status(_iso(day))


@mcp.tool
def get_daily_training_status(day: str | None = None) -> dict[str, Any]:
    return garmin().get_daily_training_status(_iso(day))


@mcp.tool
def get_four_week_load_balance(day: str | None = None) -> dict[str, Any]:
    return garmin().get_training_four_week_load_balance(_iso(day))


@mcp.tool
def get_max_metrics_latest(day: str | None = None) -> dict[str, Any]:
    return garmin().get_max_metrics_latest(_iso(day))


@mcp.tool
def get_cycling_ability(day: str | None = None) -> dict[str, Any]:
    return garmin().get_cycling_ability(_iso(day))


@mcp.tool
def get_power_curve(start_date: str, end_date: str) -> dict[str, Any]:
    return garmin().get_power_curve(start_date, end_date, sport="cycling")


@mcp.tool
def get_training_load_activities(
    start_date: str,
    end_date: str,
    activity_type: str | None = None,
) -> list[dict[str, Any]]:
    return garmin().get_training_load_activities(
        start_date, end_date, activitytype=activity_type
    )


@mcp.tool
def get_endurance_score(
    start_date: str,
    end_date: str | None = None,
    aggregation: str = "daily",
) -> dict[str, Any]:
    return garmin().get_endurance_score(start_date, end_date, aggregation=aggregation)


@mcp.tool
def get_heat_altitude_acclimation(day: str | None = None) -> dict[str, Any]:
    return garmin().get_heat_altitude_acclimation(_iso(day))


@mcp.tool
def get_cycling_stamina_curve(day: str | None = None) -> dict[str, Any]:
    return garmin().get_stamina_curve(_iso(day), sport_type="cycling")


@mcp.tool
def get_cycling_ability_range(
    start_date: str,
    end_date: str,
    aggregation: str = "DAILY",
) -> dict[str, Any]:
    return garmin().get_cycling_ability_range(
        start_date, end_date, aggregation=aggregation
    )


@mcp.tool
def get_vo2max_history(
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return garmin().get_max_metrics_range(start_date, end_date)


@mcp.tool
def get_training_history_bundle(
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Longitudinal cycling/training metrics for analysis, with independent errors per source."""
    c = garmin()
    return {
        "start_date": start_date,
        "end_date": end_date,
        "ftp_history": _safe_optional(
            c.get_functional_threshold_power_range,
            start_date,
            end_date,
            sport="CYCLING",
            aggregation="daily",
        ),
        "vo2max_history": _safe_optional(c.get_max_metrics_range, start_date, end_date),
        "cycling_ability_history": _safe_optional(
            c.get_cycling_ability_range, start_date, end_date, aggregation="DAILY"
        ),
        "power_curve": _safe_optional(
            c.get_power_curve, start_date, end_date, sport="cycling"
        ),
        "training_load_activities": _safe_optional(
            c.get_training_load_activities, start_date, end_date
        ),
        "weekly_training_status": _safe_optional(
            c.get_weekly_training_status, start_date, end_date
        ),
        "heat_altitude_history": _safe_optional(
            c.get_heat_altitude_acclimation_range,
            start_date,
            end_date,
            aggregation="daily",
        ),
        "endurance_score_history": _safe_optional(
            c.get_endurance_score,
            start_date,
            end_date,
            aggregation="daily",
        ),
    }


@mcp.tool
def get_training_analysis_bundle(day: str | None = None) -> dict[str, Any]:
    d = _iso(day)
    c = garmin()
    return {
        "date": d,
        "cycling_ftp": _safe_optional(c.get_cycling_ftp),
        "heart_rate_zones": _safe_optional(c.get_heart_rate_zones),
        "power_zones": _safe_optional(c.get_power_zones_for_sport, "CYCLING"),
        "training_status": _safe_optional(c.get_training_status, d),
        "daily_training_status": _safe_optional(c.get_daily_training_status, d),
        "four_week_load_balance": _safe_optional(c.get_training_four_week_load_balance, d),
        "max_metrics": _safe_optional(c.get_max_metrics_latest, d),
        "cycling_ability": _safe_optional(c.get_cycling_ability, d),
        "endurance_score": _safe_optional(c.get_endurance_score, d, d, aggregation="daily"),
        "heat_altitude_acclimation": _safe_optional(c.get_heat_altitude_acclimation, d),
        "cycling_stamina_curve": _safe_optional(c.get_stamina_curve, d, sport_type="cycling"),
    }


def _startup_selftest() -> None:
    """Verify Garmin auth and the detailed cycling data path without logging personal data."""
    if os.environ.get("GARMIN_STARTUP_SELFTEST") != "1":
        return
    try:
        c = garmin()
        activities = c.get_activities(0, 1)
        if not activities:
            log.info("GARMIN_SELFTEST_OK auth=true recent_activity=false")
            return
        activity_id = str(
            activities[0].get("activityId")
            or activities[0].get("activity_id")
            or activities[0].get("id")
        )
        if not activity_id or activity_id == "None":
            raise RuntimeError("Recent activity did not contain an activity id")
        detail = c.get_activity_details(activity_id, maxchart=10000, maxpoly=0)
        laps = c.get_activity_laps(activity_id)
        try:
            power_curve = c.get_activity_power_curve(activity_id)
            power_curve_ok = power_curve is not None
        except Exception:
            power_curve_ok = False
        detail_ok = isinstance(detail, dict) and bool(detail)
        laps_ok = isinstance(laps, dict) and bool(laps)
        log.info(
            "GARMIN_SELFTEST_OK auth=true recent_activity=true detail=%s laps=%s power_curve=%s",
            detail_ok,
            laps_ok,
            power_curve_ok,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("GARMIN_SELFTEST_FAILED %s", type(exc).__name__)


if __name__ == "__main__":
    _startup_selftest()
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    path = os.environ.get("MCP_PATH", "/mcp")
    log.info("Starting Garmin MCP Extended on http://%s:%s%s", host, port, path)
    mcp.run(transport="http", host=host, port=port, path=path)
