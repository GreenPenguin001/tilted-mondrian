"""Shared ATUS data construction, labels, and feature whitelist utilities."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspect_atus_design.inspect_atus_design import (
    choose_role_files,
    code_digits,
    discover_data_files,
    entropy_from_counts,
    extract_archives,
    find_col,
    json_safe,
    make_splits,
    normalize_id_series,
    parse_hour,
    read_atus_table,
    resolve_path,
    safe_extract_zip,
    tier_digits,
    write_csv,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_DIR / "data" / "raw_0324"
DEFAULT_EXTRACT_DIR = PROJECT_DIR / "data" / "extracted_0324"
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "cache"
PREP_DIR = PROJECT_DIR / "prepared_designs"
SPLIT_ORDER = ["train", "calibration", "repair", "test"]
OTHER_LABEL = "OTHER_ACTIVITY"
PIPELINE_CACHE_VERSION = "atus_pipeline_closed_world_topk"

DATA_CODE_TIER1 = {"50"}
DATA_CODE_TIER2_PREFIXES = ("50",)
SLEEP_ACTIVITY_PREFIXES = ("0101",)

RUNNER_COLS = [
    "TUCASEID",
    "TULINENO",
    "TUYEAR",
    "TUDIARYDAY",
    "TRHOLIDAY",
    "TELFS",
    "TRCHILDNUM",
    "TRNUMHOU",
    "TUACTIVITY_N",
    "TUSTARTTIM",
    "TUSTOPTIME",
    "TRCODEP",
    "TRTIER1P",
    "TRTIER2P",
    "TRTIER3P",
    "TUACTDUR24",
    "TEWHERE",
    "GEREG",
    "PEEDUCA",
    "PEHSPNON",
    "PEMARITL",
    "PEMLR",
    "PESEX",
    "PRTAGE",
    "PTDTRACE",
    "HRNUMHOU",
    "TEAGE",
    "TESEX",
    "GEDIV",
    "GEMETSTA",
    "GESTFIPS",
    "GTMETSTA",
    "HEFAMINC",
    "HEHOUSUT",
    "HETENURE",
    "HRHTYPE",
    "HRMIS",
    "HRMONTH",
    "HUFAMINC",
    "HUBUS",
    "PECERT1",
    "PECERT2",
    "PECERT3",
    "PECYC",
    "PEDIPGED",
    "PEERN",
    "PEERNH1O",
    "PEERNHRO",
    "PEERNHRY",
    "PEERNPER",
    "PEERNRT",
    "PEERNWKP",
    "PEGR6COR",
    "PEGRPROF",
    "PEHGCOMP",
    "PEHRFTPT",
    "PEHRUSL1",
    "PEHRUSL2",
    "PEHRUSLT",
    "PEIO1COW",
    "PEIO1ICD",
    "PEIO1OCD",
    "PEMJOT",
    "PESCHENR",
    "PESCHFT",
    "PESCHLVL",
    "PRCITSHIP",
    "PRCOW1",
    "PRDTCOW1",
    "PRDTIND1",
    "PRDTOCC1",
    "PRFTLF",
    "PRHRUSL",
    "PRMJIND1",
    "PRMJOCC1",
    "PRMJOCGR",
    "PRNMCHLD",
    "PRWKSTAT",
    "PRWKSCH",
    "TEERNH1O",
    "TEERNHRO",
    "TEERNHRY",
    "TEERNPER",
    "TEERNRT",
    "TEERNWKP",
    "TEHRFTPT",
    "TEHRUSL1",
    "TEHRUSL2",
    "TEHRUSLT",
    "TEIO1COW",
    "TEIO1ICD",
    "TEIO1OCD",
    "TEMJOT",
    "TESCHENR",
    "TESCHFT",
    "TESCHLVL",
    "TESPEMPNOT",
    "TESPUHRS",
    "TRDPFTPT",
    "TRDTIND1",
    "TRDTOCC1",
    "TRERNHLY",
    "TRERNWA",
    "TRHHCHILD",
    "TRIMIND1",
    "TRMJIND1",
    "TRMJOCC1",
    "TRMJOCGR",
    "TRNHHCHILD",
    "TROHHCHILD",
    "TRSPFTPT",
    "TRSPPRES",
    "TRYHHCHILD",
    "TUBUS",
    "TUMONTH",
    "TUSPABS",
    "TUSPUSFT",
    "TUSPWK",
]

FALLBACK_TIER2_NAMES = {
    "0101": "Sleeping",
    "0102": "Grooming",
    "0103": "Health-related self care",
    "0201": "Housework",
    "0202": "Food and drink preparation",
    "0206": "Household management",
    "0209": "Lawn, garden, and houseplants",
    "0301": "Caring for household children",
    "0501": "Working",
    "0701": "Consumer purchases",
    "1101": "Eating and drinking",
    "1201": "Socializing and communicating",
    "1203": "Watching television and movies",
    "1301": "Sports, exercise, and recreation",
    "1401": "Religious and spiritual activities",
    "1803": "Travel related to work",
    "1804": "Travel related to education",
    "1805": "Travel related to consumer purchases",
    "1807": "Travel related to eating and drinking",
    "1811": "Travel related to socializing and leisure",
    "1812": "Travel related to sports, exercise, and recreation",
    "1813": "Travel related to religious and spiritual activities",
    "5001": "Data code: unable to code",
}

FEATURE_WHITELIST = {
    "engineered_features": [
        {"feature": "start_hour", "source_columns": ["activity:TUSTARTTIM"], "type": "numeric", "allowed_reason": "activity start time source only"},
        {"feature": "start_hour_sin", "source_columns": ["activity:TUSTARTTIM"], "type": "numeric", "allowed_reason": "cyclic activity start hour source"},
        {"feature": "start_hour_cos", "source_columns": ["activity:TUSTARTTIM"], "type": "numeric", "allowed_reason": "cyclic activity start hour source"},
        {"feature": "start_time_bucket", "source_columns": ["activity:TUSTARTTIM"], "type": "categorical", "allowed_reason": "coarse start time bucket/daypart source"},
        {"feature": "diary_day", "source_columns": ["respondent:TUDIARYDAY"], "type": "categorical", "allowed_reason": "diary day context"},
        {"feature": "is_weekend", "source_columns": ["respondent:TUDIARYDAY"], "type": "binary", "allowed_reason": "derived from diary day"},
        {"feature": "is_holiday", "source_columns": ["respondent:TRHOLIDAY"], "type": "binary", "allowed_reason": "diary holiday context"},
        {"feature": "year", "source_columns": ["respondent:TUYEAR"], "type": "numeric", "allowed_reason": "survey year"},
        {"feature": "age", "source_columns": ["atus_cps:PRTAGE"], "type": "numeric", "allowed_reason": "respondent age"},
        {"feature": "education", "source_columns": ["atus_cps:PEEDUCA"], "type": "categorical", "allowed_reason": "respondent education"},
        {"feature": "employment_status", "source_columns": ["respondent:TELFS", "atus_cps:PEMLR"], "type": "categorical", "allowed_reason": "pre-activity labor force context"},
        {"feature": "marital_status", "source_columns": ["atus_cps:PEMARITL"], "type": "categorical", "allowed_reason": "respondent marital context"},
        {"feature": "household_child_context", "source_columns": ["roster:TEAGE"], "type": "categorical", "allowed_reason": "household child context derived from roster ages"},
        {"feature": "household_size", "source_columns": ["roster:TUCASEID"], "type": "numeric", "allowed_reason": "household roster size"},
        {"feature": "region", "source_columns": ["atus_cps:GEREG"], "type": "categorical", "allowed_reason": "Census region"},
    ],
    "forbidden_sources": [
        "activity code",
        "tier activity labels",
        "activity description",
        "duration",
        "stop time",
        "where/location",
        "who-with",
        "activity summary variables",
        "TRTO/TRT activity-derived summaries",
        "G variables",
        "true Y",
        "true Z for propensity H",
        "set size / alpha / threshold / conformal outputs",
    ],
    "propensity_h_rule": {
        "labels_ge_15": ["top1_probability", "top2_probability", "top3_probability", "entropy", "margin"],
        "labels_lt_12": ["p_max", "p_second", "margin", "entropy"],
        "excluded": ["true_G", "true_Y", "true_Z", "set_size", "alpha", "threshold"],
    },
}

STRONG_WHERE_FEATURE = {
    "feature": "where_context",
    "source_columns": ["activity:TEWHERE"],
    "type": "categorical",
    "allowed_reason": "strong-context ablation only; episode where/location can proxy activity and is excluded from the primary safe feature set",
}

SAFE_PLUS_CONTEXT_FEATURES = [
    {"feature": "ctx_occupation_major", "source_columns": ["respondent:TRMJOCC1", "atus_cps:PRMJOCC1"], "type": "categorical", "allowed_reason": "pre-activity respondent occupation major code", "top_n": 30},
    {"feature": "ctx_occupation_group", "source_columns": ["respondent:TRMJOCGR", "atus_cps:PRMJOCGR"], "type": "categorical", "allowed_reason": "pre-activity respondent occupation group", "top_n": 20},
    {"feature": "ctx_occupation_detailed", "source_columns": ["respondent:TRDTOCC1", "atus_cps:PRDTOCC1"], "type": "categorical", "allowed_reason": "pre-activity detailed occupation code collapsed to train top categories", "top_n": 40},
    {"feature": "ctx_industry_major", "source_columns": ["respondent:TRMJIND1", "atus_cps:PRMJIND1"], "type": "categorical", "allowed_reason": "pre-activity respondent industry major code", "top_n": 25},
    {"feature": "ctx_industry_intermediate", "source_columns": ["respondent:TRIMIND1"], "type": "categorical", "allowed_reason": "pre-activity respondent industry intermediate code", "top_n": 30},
    {"feature": "ctx_industry_detailed", "source_columns": ["respondent:TRDTIND1", "atus_cps:PRDTIND1"], "type": "categorical", "allowed_reason": "pre-activity detailed industry code collapsed to train top categories", "top_n": 40},
    {"feature": "ctx_class_worker", "source_columns": ["respondent:TEIO1COW", "atus_cps:PEIO1COW", "atus_cps:PRCOW1", "atus_cps:PRDTCOW1"], "type": "categorical", "allowed_reason": "class of worker from respondent/CPS context", "top_n": 20},
    {"feature": "ctx_full_part_time", "source_columns": ["respondent:TRDPFTPT", "respondent:TEHRFTPT", "atus_cps:PEHRFTPT", "atus_cps:PRFTLF"], "type": "categorical", "allowed_reason": "full-time/part-time labor-force context", "top_n": 12},
    {"feature": "ctx_work_status_detail", "source_columns": ["atus_cps:PRWKSTAT", "atus_cps:PRWKSCH"], "type": "categorical", "allowed_reason": "CPS work-status detail", "top_n": 20},
    {"feature": "ctx_usual_hours_total", "source_columns": ["respondent:TEHRUSLT", "atus_cps:PEHRUSLT", "atus_cps:PRHRUSL"], "type": "numeric", "allowed_reason": "usual weekly work hours before diary activity"},
    {"feature": "ctx_usual_hours_main", "source_columns": ["respondent:TEHRUSL1", "atus_cps:PEHRUSL1"], "type": "numeric", "allowed_reason": "usual weekly hours at main job"},
    {"feature": "ctx_more_than_one_job", "source_columns": ["respondent:TEMJOT", "atus_cps:PEMJOT"], "type": "categorical", "allowed_reason": "multiple-job status from respondent/CPS context", "top_n": 8},
    {"feature": "ctx_family_income", "source_columns": ["atus_cps:HEFAMINC", "atus_cps:HUFAMINC"], "type": "categorical", "allowed_reason": "household/family income bracket from CPS", "top_n": 25},
    {"feature": "ctx_hourly_paid", "source_columns": ["respondent:TEERNRT", "atus_cps:PEERNRT"], "type": "categorical", "allowed_reason": "hourly-paid indicator from respondent/CPS context", "top_n": 8},
    {"feature": "ctx_pay_frequency", "source_columns": ["respondent:TEERNPER", "atus_cps:PEERNPER"], "type": "categorical", "allowed_reason": "earnings reporting period from respondent/CPS context", "top_n": 12},
    {"feature": "ctx_weekly_earnings", "source_columns": ["respondent:TRERNWA", "atus_cps:PRERNWA", "atus_cps:PEERN"], "type": "numeric", "allowed_reason": "pre-activity weekly earnings context"},
    {"feature": "ctx_hourly_earnings", "source_columns": ["respondent:TRERNHLY", "atus_cps:PRERNHLY", "respondent:TEERNH1O", "atus_cps:PEERNH1O"], "type": "numeric", "allowed_reason": "pre-activity hourly earnings context"},
    {"feature": "ctx_spouse_present", "source_columns": ["respondent:TRSPPRES"], "type": "categorical", "allowed_reason": "spouse/partner presence in household", "top_n": 8},
    {"feature": "ctx_spouse_employment", "source_columns": ["respondent:TESPEMPNOT", "respondent:TUSPWK", "respondent:TUSPABS"], "type": "categorical", "allowed_reason": "spouse/partner employment context", "top_n": 12},
    {"feature": "ctx_spouse_full_time", "source_columns": ["respondent:TRSPFTPT", "respondent:TUSPUSFT"], "type": "categorical", "allowed_reason": "spouse/partner full-time context", "top_n": 8},
    {"feature": "ctx_spouse_usual_hours", "source_columns": ["respondent:TESPUHRS"], "type": "numeric", "allowed_reason": "spouse/partner usual weekly hours"},
    {"feature": "ctx_school_enrolled", "source_columns": ["respondent:TESCHENR", "atus_cps:PESCHENR"], "type": "categorical", "allowed_reason": "school enrollment context", "top_n": 8},
    {"feature": "ctx_school_full_time", "source_columns": ["respondent:TESCHFT", "atus_cps:PESCHFT"], "type": "categorical", "allowed_reason": "full-time/part-time school enrollment", "top_n": 8},
    {"feature": "ctx_school_level", "source_columns": ["respondent:TESCHLVL", "atus_cps:PESCHLVL"], "type": "categorical", "allowed_reason": "school level context", "top_n": 8},
    {"feature": "ctx_education_detail", "source_columns": ["atus_cps:PECYC", "atus_cps:PEDIPGED", "atus_cps:PEGR6COR", "atus_cps:PEGRPROF", "atus_cps:PEHGCOMP"], "type": "categorical", "allowed_reason": "additional education detail from CPS", "top_n": 20},
    {"feature": "ctx_num_children_under18", "source_columns": ["respondent:TRCHILDNUM", "atus_cps:PRNMCHLD"], "type": "numeric", "allowed_reason": "number of children under 18 from respondent/CPS context"},
    {"feature": "ctx_youngest_child_age", "source_columns": ["respondent:TRYHHCHILD"], "type": "numeric", "allowed_reason": "age of youngest household child under 18"},
    {"feature": "ctx_household_children_presence", "source_columns": ["respondent:TRHHCHILD", "respondent:TROHHCHILD", "respondent:TRNHHCHILD"], "type": "categorical", "allowed_reason": "household/nonhousehold child presence context", "top_n": 12},
    {"feature": "ctx_household_type", "source_columns": ["atus_cps:HRHTYPE"], "type": "categorical", "allowed_reason": "household type from CPS", "top_n": 20},
    {"feature": "ctx_housing_tenure", "source_columns": ["atus_cps:HETENURE"], "type": "categorical", "allowed_reason": "housing tenure from CPS", "top_n": 12},
    {"feature": "ctx_housing_unit_type", "source_columns": ["atus_cps:HEHOUSUT"], "type": "categorical", "allowed_reason": "housing-unit type from CPS", "top_n": 20},
    {"feature": "ctx_division", "source_columns": ["atus_cps:GEDIV"], "type": "categorical", "allowed_reason": "Census division from CPS", "top_n": 12},
    {"feature": "ctx_state", "source_columns": ["atus_cps:GESTFIPS"], "type": "categorical", "allowed_reason": "state FIPS code from CPS; coarse geography context", "top_n": 60},
    {"feature": "ctx_metro_status", "source_columns": ["atus_cps:GTMETSTA", "atus_cps:GEMETSTA"], "type": "categorical", "allowed_reason": "metropolitan status from CPS", "top_n": 8},
    {"feature": "ctx_month", "source_columns": ["respondent:TUMONTH", "atus_cps:HRMONTH"], "type": "categorical", "allowed_reason": "interview/diary month context", "top_n": 15},
    {"feature": "ctx_household_business", "source_columns": ["respondent:TUBUS", "atus_cps:HUBUS"], "type": "categorical", "allowed_reason": "household business/farm ownership context", "top_n": 8},
]

FORBIDDEN_PATTERN_REASONS = [
    (re.compile(r"TRCODE|TIER|ACTIVITY_CODE|Y_LABEL|TIER2_CODE|TIER2_NAME", re.I), "activity code/tier/Y label"),
    (re.compile(r"DUR|STOP|WHERE|WHO", re.I), "duration/stop/where/who proxy"),
    (re.compile(r"TRTO|TRTCC|TRTCOC|TRTHH|TRTNO|TRTALONE|SUMMARY", re.I), "activity-derived summary"),
    (re.compile(r"SEX_G|RACE_G|TESEX|PESEX|PTDTRACE|PEHSPNON", re.I), "G variable"),
    (re.compile(r"SET_SIZE|ALPHA|THRESHOLD|CONFORMAL", re.I), "conformal output"),
]


def feature_whitelist_for(feature_mode: str) -> dict[str, Any]:
    if feature_mode not in {"safe", "strong_where", "safe_plus_context"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    whitelist = {
        "feature_mode": feature_mode,
        "engineered_features": [dict(row) for row in FEATURE_WHITELIST["engineered_features"]],
        "forbidden_sources": list(FEATURE_WHITELIST["forbidden_sources"]),
        "propensity_h_rule": dict(FEATURE_WHITELIST["propensity_h_rule"]),
    }
    if feature_mode == "strong_where":
        whitelist["engineered_features"].append(dict(STRONG_WHERE_FEATURE))
        whitelist["forbidden_sources"] = [x for x in whitelist["forbidden_sources"] if x != "where/location"]
        whitelist["strong_context_ablation_sources"] = ["where/location"]
        whitelist["strong_context_ablation"] = True
    elif feature_mode == "safe_plus_context":
        whitelist["engineered_features"].extend(dict(row) for row in SAFE_PLUS_CONTEXT_FEATURES)
        whitelist["safe_plus_context"] = True
        whitelist["safe_plus_context_rule"] = "safe X plus non-episode-proxy respondent/CPS/roster context only; where/who/duration/stop/activity-derived variables remain excluded"
        whitelist["strong_context_ablation"] = False
    else:
        whitelist["strong_context_ablation"] = False
    return whitelist


def slug_label(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def normalize_code(value: Any, width: int) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    return match.group(1).zfill(width)


def load_activity_lexicon(lexicon_path: Path | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    if lexicon_path is None:
        return {}, {
            "lexicon_mapping_status": "fallback_builtin_no_lexicon_path",
            "lexicon_path": None,
            "mapped_codes": 0,
            "notes": "No official lexicon path was provided; using built-in fallback names where available and generic names otherwise.",
        }
    path = resolve_path(Path(lexicon_path))
    if not path.exists():
        return {}, {
            "lexicon_mapping_status": "fallback_lexicon_path_missing",
            "lexicon_path": str(path),
            "mapped_codes": 0,
            "notes": "Provided lexicon path does not exist.",
        }
    if path.suffix.lower() == ".pdf":
        return {}, {
            "lexicon_mapping_status": "fallback_pdf_lexicon_not_parsed",
            "lexicon_path": str(path),
            "mapped_codes": 0,
            "notes": "PDF lexicon parsing is not enabled in this local pipeline. Provide CSV/TSV/TXT with activity code and label columns for official names.",
        }
    try:
        if path.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, low_memory=False)
            code_col = find_col(df.columns, ["code", "activity_code", "TRCODEP", "tier2_code"], contains_any=["CODE"])
            name_col = find_col(df.columns, ["name", "activity_name", "label", "description"], contains_any=["NAME", "LABEL", "DESCRIPTION", "TITLE"])
            if code_col is None or name_col is None:
                raise ValueError("could not infer code/name columns")
            mapping = {}
            for _, row in df[[code_col, name_col]].dropna().iterrows():
                code = normalize_code(row[code_col], 4 if len(re.sub(r"\D", "", str(row[code_col]))) <= 4 else 6)
                if code is None:
                    continue
                mapping[code[:4]] = str(row[name_col]).strip()
            return mapping, {
                "lexicon_mapping_status": "official_lexicon_csv_loaded",
                "lexicon_path": str(path),
                "mapped_codes": len(mapping),
                "notes": "Loaded user-provided CSV/TSV lexicon.",
            }
        text = path.read_text(encoding="utf-8", errors="replace")
        mapping = {}
        for line in text.splitlines():
            match = re.match(r"^\s*(\d{4,6})\s+(.+?)\s*$", line)
            if match:
                mapping[match.group(1)[:4]] = match.group(2).strip()
        return mapping, {
            "lexicon_mapping_status": "official_lexicon_text_loaded" if mapping else "fallback_text_lexicon_no_matches",
            "lexicon_path": str(path),
            "mapped_codes": len(mapping),
            "notes": "Parsed text lexicon with leading 4-6 digit code lines.",
        }
    except Exception as exc:  # noqa: BLE001
        return {}, {
            "lexicon_mapping_status": "fallback_lexicon_parse_failed",
            "lexicon_path": str(path),
            "mapped_codes": 0,
            "notes": f"{type(exc).__name__}: {exc}",
        }


def tier2_name_and_source(code: Any, official_map: dict[str, str]) -> tuple[str, str]:
    code_text = normalize_code(code, 4)
    if code_text is None:
        return "Unknown activity", "manual"
    if code_text in official_map:
        return official_map[code_text], "official"
    if code_text in FALLBACK_TIER2_NAMES:
        return FALLBACK_TIER2_NAMES[code_text], "fallback"
    tier1 = code_text[:2]
    tier1_names = {
        "01": "Personal care",
        "02": "Household activities",
        "03": "Caring for household members",
        "04": "Caring for nonhousehold members",
        "05": "Work and work-related activities",
        "06": "Education",
        "07": "Consumer purchases",
        "08": "Professional and personal care services",
        "09": "Household services",
        "10": "Government services and civic obligations",
        "11": "Eating and drinking",
        "12": "Socializing, relaxing, and leisure",
        "13": "Sports, exercise, and recreation",
        "14": "Religious and spiritual activities",
        "15": "Volunteer activities",
        "16": "Telephone calls",
        "18": "Traveling",
        "50": "Data code",
    }
    return f"{tier1_names.get(tier1, 'Activity')} tier-2 code {code_text}", "manual"


def tier2_name(code: Any, official_map: dict[str, str]) -> str:
    return tier2_name_and_source(code, official_map)[0]


def map_sex(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"1", "1.0", "male", "m"}:
        return "male"
    if text in {"2", "2.0", "female", "f"}:
        return "female"
    return pd.NA


def map_employment(value: Any) -> str:
    if pd.isna(value):
        return "unknown_or_other"
    text = str(value).strip().lower()
    numeric = re.sub(r"\.0$", "", text)
    if numeric in {"1", "2"}:
        return "employed"
    if numeric in {"3", "4"}:
        return "unemployed"
    if numeric in {"5", "6", "7"}:
        return "not_in_labor_force"
    if "unemploy" in text or "layoff" in text or "looking" in text:
        return "unemployed"
    if "not in labor" in text or "nilf" in text:
        return "not_in_labor_force"
    if "employ" in text or "at work" in text or "absent" in text:
        return "employed"
    return "unknown_or_other"


def is_hispanic(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    numeric = re.sub(r"\.0$", "", str(value).strip().lower())
    if numeric == "1":
        return True
    if numeric == "2":
        return False
    return None


def race_white_black(value: Any) -> str | None:
    if pd.isna(value):
        return None
    numeric = re.sub(r"\.0$", "", str(value).strip().lower())
    if numeric == "1":
        return "white"
    if numeric == "2":
        return "black"
    text = str(value).strip().lower()
    if "white" in text:
        return "white"
    if "black" in text or "african" in text:
        return "black"
    return None


def map_race_group(race_value: Any, hisp_value: Any) -> str | pd.NA:
    race = race_white_black(race_value)
    if race is None:
        return pd.NA
    hisp = is_hispanic(hisp_value)
    if hisp is True:
        return pd.NA
    if race == "white":
        return "non_hispanic_white"
    if race == "black":
        return "non_hispanic_black"
    return pd.NA


def start_bucket(hour: Any) -> str:
    if pd.isna(hour):
        return "unknown_start"
    h = float(hour)
    if 0 <= h < 6:
        return "night"
    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    if 18 <= h < 24:
        return "evening"
    return "unknown_start"


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TUCASEID" in out.columns:
        out["TUCASEID"] = normalize_id_series(out["TUCASEID"])
    if "TULINENO" in out.columns:
        out["TULINENO"] = normalize_id_series(out["TULINENO"])
    return out


def discover_and_read_tables(raw_dir: Path, extract_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    raw_dir = resolve_path(raw_dir)
    extract_dir = resolve_path(extract_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_archives(raw_dir, extract_dir)
    inventory = discover_data_files(raw_dir, extract_dir)
    selected = choose_role_files(inventory)
    missing = [role for role in ["respondent", "activity"] if role not in selected]
    if missing:
        raise FileNotFoundError(f"Missing required ATUS files after local discovery: {missing}")
    tables: dict[str, pd.DataFrame] = {}
    read_status = []
    for role in ["respondent", "activity", "atus_cps", "roster"]:
        path = selected.get(role)
        if path is None:
            continue
        df, info = read_atus_table(path, role, RUNNER_COLS)
        read_status.append(info)
        if df is None:
            raise RuntimeError(f"Failed to read {role}: {info}")
        tables[role] = normalize_key_columns(df)
    meta = {
        "selected_files": {role: str(path) for role, path in selected.items()},
        "read_status": read_status,
        "inventory_rows": int(len(inventory)),
    }
    return tables, meta


def respondent_context(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    respondent = tables["respondent"].copy()
    respondent = respondent.drop_duplicates("TUCASEID", keep="first")
    respondent_keep = [
        "TUCASEID",
        "TULINENO",
        "TUYEAR",
        "TUDIARYDAY",
        "TRHOLIDAY",
        "TELFS",
        "TRCHILDNUM",
        "TRNUMHOU",
        "TESPUHRS",
        "TEERNH1O",
        "TEERNHRO",
        "TEERNHRY",
        "TEERNPER",
        "TEERNRT",
        "TEERNWKP",
        "TEHRFTPT",
        "TEHRUSL1",
        "TEHRUSL2",
        "TEHRUSLT",
        "TEIO1COW",
        "TEIO1ICD",
        "TEIO1OCD",
        "TEMJOT",
        "TESCHENR",
        "TESCHFT",
        "TESCHLVL",
        "TESPEMPNOT",
        "TRDPFTPT",
        "TRDTIND1",
        "TRDTOCC1",
        "TRERNHLY",
        "TRERNWA",
        "TRHHCHILD",
        "TRIMIND1",
        "TRMJIND1",
        "TRMJOCC1",
        "TRMJOCGR",
        "TRNHHCHILD",
        "TROHHCHILD",
        "TRSPFTPT",
        "TRSPPRES",
        "TRYHHCHILD",
        "TUBUS",
        "TUMONTH",
        "TUSPABS",
        "TUSPUSFT",
        "TUSPWK",
    ]
    context = respondent[[c for c in respondent_keep if c in respondent.columns]].copy()

    cps = tables.get("atus_cps")
    if cps is not None and "TULINENO" in context.columns and "TULINENO" in cps.columns:
        cps_keep = [
            "TUCASEID",
            "TULINENO",
            "GEREG",
            "PEEDUCA",
            "PEHSPNON",
            "PEMARITL",
            "PEMLR",
            "PESEX",
            "PRTAGE",
            "PTDTRACE",
            "HRNUMHOU",
            "GEDIV",
            "GEMETSTA",
            "GESTFIPS",
            "GTMETSTA",
            "HEFAMINC",
            "HEHOUSUT",
            "HETENURE",
            "HRHTYPE",
            "HRMIS",
            "HRMONTH",
            "HUFAMINC",
            "HUBUS",
            "PECERT1",
            "PECERT2",
            "PECERT3",
            "PECYC",
            "PEDIPGED",
            "PEERN",
            "PEERNH1O",
            "PEERNHRO",
            "PEERNHRY",
            "PEERNPER",
            "PEERNRT",
            "PEERNWKP",
            "PEGR6COR",
            "PEGRPROF",
            "PEHGCOMP",
            "PEHRFTPT",
            "PEHRUSL1",
            "PEHRUSL2",
            "PEHRUSLT",
            "PEIO1COW",
            "PEIO1ICD",
            "PEIO1OCD",
            "PEMJOT",
            "PESCHENR",
            "PESCHFT",
            "PESCHLVL",
            "PRCITSHIP",
            "PRCOW1",
            "PRDTCOW1",
            "PRDTIND1",
            "PRDTOCC1",
            "PRFTLF",
            "PRHRUSL",
            "PRMJIND1",
            "PRMJOCC1",
            "PRMJOCGR",
            "PRNMCHLD",
            "PRWKSTAT",
            "PRWKSCH",
        ]
        cps_cols = [c for c in cps_keep if c in cps.columns]
        cps_resp = context[["TUCASEID", "TULINENO"]].merge(cps[cps_cols], on=["TUCASEID", "TULINENO"], how="left")
        if cps_resp["PESEX"].notna().sum() if "PESEX" in cps_resp.columns else 0:
            context = context.merge(cps_resp.drop(columns=["TULINENO"], errors="ignore"), on="TUCASEID", how="left", suffixes=("", "_cps"))
        else:
            cps_first = cps[cps_cols].drop_duplicates("TUCASEID", keep="first").drop(columns=["TULINENO"], errors="ignore")
            context = context.merge(cps_first, on="TUCASEID", how="left", suffixes=("", "_cps"))

    roster = tables.get("roster")
    if roster is not None:
        roster = roster.copy()
        if "TEAGE" in roster.columns:
            tmp = roster[["TUCASEID", "TEAGE"]].copy()
            tmp["age_num"] = pd.to_numeric(tmp["TEAGE"], errors="coerce")
            hh = tmp.groupby("TUCASEID")["age_num"].agg(
                household_size="count",
                has_under_6=lambda x: bool((x < 6).any()),
                has_6_17=lambda x: bool(((x >= 6) & (x <= 17)).any()),
            ).reset_index()
            hh["household_child_context"] = "no_child"
            hh.loc[hh["has_6_17"], "household_child_context"] = "has_child_6_17"
            hh.loc[hh["has_under_6"], "household_child_context"] = "has_child_under_6"
            context = context.merge(hh[["TUCASEID", "household_size", "household_child_context"]], on="TUCASEID", how="left")
        if "TESEX" in roster.columns and "TULINENO" in roster.columns and "TULINENO" in context.columns:
            rost_sex = context[["TUCASEID", "TULINENO"]].merge(roster[["TUCASEID", "TULINENO", "TESEX"]], on=["TUCASEID", "TULINENO"], how="left")
            context = context.merge(rost_sex[["TUCASEID", "TESEX"]], on="TUCASEID", how="left", suffixes=("", "_roster"))

    sex_source = context["TESEX"] if "TESEX" in context.columns and context["TESEX"].notna().any() else context.get("PESEX", pd.Series(pd.NA, index=context.index))
    context["sex_g"] = sex_source.map(map_sex).astype("string")
    context["race_g"] = [map_race_group(r, h) for r, h in zip(context.get("PTDTRACE", pd.Series(pd.NA, index=context.index)), context.get("PEHSPNON", pd.Series(pd.NA, index=context.index)))]
    context["race_g"] = context["race_g"].astype("string")
    emp_source = context["TELFS"] if "TELFS" in context.columns and context["TELFS"].notna().any() else context.get("PEMLR", pd.Series(pd.NA, index=context.index))
    context["employment_z"] = emp_source.map(map_employment).astype("string")
    context["employment_status"] = context["employment_z"].astype("string")
    context["household_z"] = context.get("household_child_context", pd.Series(pd.NA, index=context.index)).astype("string")
    return context


def make_y_label(tier2_code: Any, tier2_name_value: Any) -> str | pd.NA:
    if pd.isna(tier2_code):
        return pd.NA
    return f"tier2_{str(tier2_code)}__{slug_label(str(tier2_name_value))}"


def build_base_examples(
    raw_dir: Path,
    extract_dir: Path,
    seed: int,
    unit: str,
    exclude_data_code_activities: bool,
    lexicon_path: Path | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    cache_key = f"{PIPELINE_CACHE_VERSION}_{unit}_seed{seed}_exclude_data{int(exclude_data_code_activities)}_lex{bool(lexicon_path)}.pkl"
    cache_path = DEFAULT_CACHE_DIR / cache_key
    if use_cache and cache_path.exists():
        payload = pd.read_pickle(cache_path)
        return payload["examples"], payload["metadata"], payload["excluded_codes"]

    official_map, lexicon_meta = load_activity_lexicon(lexicon_path)
    tables, read_meta = discover_and_read_tables(raw_dir, extract_dir)
    activity = tables["activity"].copy()
    context = respondent_context(tables)
    activity["TUCASEID"] = normalize_id_series(activity["TUCASEID"])
    code_col = find_col(activity.columns, ["TRCODEP"], contains_any=["TRCODE"])
    tier1_col = find_col(activity.columns, ["TRTIER1P", "TRTIER1"], contains_any=["TIER1"])
    tier2_col = find_col(activity.columns, ["TRTIER2P", "TRTIER2"], contains_any=["TIER2"])
    start_col = find_col(activity.columns, ["TUSTARTTIM"], contains_any=["START"])

    if code_col:
        activity["activity_code"] = code_digits(activity[code_col], 6)
    else:
        activity["activity_code"] = pd.NA
    if tier1_col:
        activity["tier1_code"] = tier_digits(activity[tier1_col], 2)
    else:
        activity["tier1_code"] = activity["activity_code"].astype("string").str.slice(0, 2)
    if tier2_col:
        activity["tier2_code"] = tier_digits(activity[tier2_col], 4)
    else:
        activity["tier2_code"] = activity["activity_code"].astype("string").str.slice(0, 4)
    tier2_mapped = activity["tier2_code"].map(lambda x: tier2_name_and_source(x, official_map))
    activity["tier2_name"] = tier2_mapped.map(lambda x: x[0]).astype("string")
    activity["tier2_mapping_source"] = tier2_mapped.map(lambda x: x[1]).astype("string")
    activity["y_label_raw"] = [make_y_label(c, n) for c, n in zip(activity["tier2_code"], activity["tier2_name"])]
    activity["is_sleep"] = activity["activity_code"].astype("string").str.startswith(SLEEP_ACTIVITY_PREFIXES, na=False)
    activity["is_data_code_activity"] = (
        activity["tier1_code"].astype("string").isin(DATA_CODE_TIER1)
        | activity["tier2_code"].astype("string").str.startswith(DATA_CODE_TIER2_PREFIXES, na=False)
        | activity["activity_code"].astype("string").str.startswith(DATA_CODE_TIER2_PREFIXES, na=False)
    )
    if start_col:
        hours = activity[start_col].map(parse_hour)
    else:
        hours = pd.Series(np.nan, index=activity.index)
    activity["start_hour"] = pd.to_numeric(hours, errors="coerce")
    activity["start_hour_sin"] = np.sin(2 * np.pi * activity["start_hour"] / 24.0)
    activity["start_hour_cos"] = np.cos(2 * np.pi * activity["start_hour"] / 24.0)
    activity["start_time_bucket"] = activity["start_hour"].map(start_bucket).astype("string")

    excluded_codes = (
        activity.loc[activity["is_data_code_activity"], ["activity_code", "tier1_code", "tier2_code", "tier2_name"]]
        .fillna("MISSING")
        .groupby(["activity_code", "tier1_code", "tier2_code", "tier2_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    excluded_codes["action"] = "excluded" if exclude_data_code_activities else "mapped_to_other_if_not_topk"

    eligible = activity[~activity["is_sleep"]].copy()
    if exclude_data_code_activities:
        eligible = eligible[~eligible["is_data_code_activity"]].copy()
    eligible = eligible.dropna(subset=["TUCASEID", "tier2_code", "y_label_raw"])
    eligible = eligible.merge(context, on="TUCASEID", how="left")
    split_map = make_splits(eligible["TUCASEID"], seed)
    eligible["split"] = eligible["TUCASEID"].astype("string").map(split_map).astype("string")

    if unit == "one_episode_per_respondent":
        examples = (
            eligible.sample(frac=1.0, random_state=seed)
            .drop_duplicates("TUCASEID", keep="first")
            .sort_values(["split", "TUCASEID"])
            .reset_index(drop=True)
        )
    elif unit == "all_eligible_episodes":
        examples = eligible.reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported ATUS unit: {unit}")

    examples["diary_day"] = examples.get("TUDIARYDAY", pd.Series(pd.NA, index=examples.index)).astype("string")
    diary_num = pd.to_numeric(examples.get("TUDIARYDAY", pd.Series(np.nan, index=examples.index)), errors="coerce")
    examples["is_weekend"] = diary_num.isin([1, 7]).astype(int)
    examples["is_holiday"] = pd.to_numeric(examples.get("TRHOLIDAY", pd.Series(0, index=examples.index)), errors="coerce").fillna(0).astype(int)
    examples["year"] = pd.to_numeric(examples.get("TUYEAR", pd.Series(np.nan, index=examples.index)), errors="coerce")
    examples["age"] = pd.to_numeric(examples.get("PRTAGE", pd.Series(np.nan, index=examples.index)), errors="coerce")
    examples["education"] = examples.get("PEEDUCA", pd.Series(pd.NA, index=examples.index)).astype("string")
    examples["marital_status"] = examples.get("PEMARITL", pd.Series(pd.NA, index=examples.index)).astype("string")
    examples["household_child_context"] = examples.get("household_child_context", pd.Series(pd.NA, index=examples.index)).fillna("unknown_or_other").astype("string")
    examples["household_size"] = pd.to_numeric(examples.get("household_size", pd.Series(np.nan, index=examples.index)), errors="coerce")
    examples["region"] = examples.get("GEREG", pd.Series(pd.NA, index=examples.index)).astype("string")

    metadata = {
        **read_meta,
        **lexicon_meta,
        "unit": unit,
        "seed": seed,
        "exclude_data_code_activities": exclude_data_code_activities,
        "sleep_exclusion_rule": "TRCODEP normalized to six digits and prefix 0101",
        "data_code_activity_rule": "tier1_code == 50 or tier2/activity code starts with 50",
        "data_code_activity_rows_total": int(activity["is_data_code_activity"].sum()),
        "data_code_activity_rows_after_sleep_filter": int((~activity["is_sleep"] & activity["is_data_code_activity"]).sum()),
        "eligible_examples_before_unit_sampling": int(len(eligible)),
        "examples_after_unit_sampling": int(len(examples)),
        "cache_path": str(cache_path),
        "cache_created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if use_cache:
        pd.to_pickle({"examples": examples, "metadata": metadata, "excluded_codes": excluded_codes}, cache_path)
    return examples, metadata, excluded_codes


def is_closed_world_y_mode(y_mode: str) -> bool:
    return y_mode in {"closed_world_top20", "closed_world_topk"}


def apply_train_topk_labels(
    examples: pd.DataFrame,
    top_k: int,
    y_mode: str = "top20_with_other",
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if y_mode not in {"top20_with_other", "closed_world_top20", "closed_world_topk"}:
        raise ValueError(f"Unsupported y_mode: {y_mode}")
    out = examples.copy()
    train = out[out["split"].eq("train")]
    train_counts = train["y_label_raw"].dropna().astype("string").value_counts()
    vocab = train_counts.head(int(top_k)).index.tolist()
    vocab_set = set(vocab)
    label_rows = []
    top_info = (
        train.loc[train["y_label_raw"].isin(vocab_set), ["y_label_raw", "tier2_code", "tier2_name", "tier2_mapping_source"]]
        .drop_duplicates("y_label_raw")
        .set_index("y_label_raw", drop=False)
    )
    for rank, label in enumerate(vocab, start=1):
        row = top_info.loc[label] if label in top_info.index else {}
        label_rows.append({
            "rank": rank,
            "tier2_code": str(row.get("tier2_code", "")) if hasattr(row, "get") else "",
            "tier2_name": str(row.get("tier2_name", "")) if hasattr(row, "get") else "",
            "y_label": str(label),
            "mapping_source": str(row.get("tier2_mapping_source", "")) if hasattr(row, "get") else "",
            "train_count": int(train_counts.get(label, 0)),
        })

    split_filter_rows = []
    if is_closed_world_y_mode(y_mode):
        before = out.groupby("split").size().reindex(SPLIT_ORDER, fill_value=0)
        out["is_train_topk_label"] = out["y_label_raw"].isin(vocab_set)
        out = out[out["is_train_topk_label"]].copy()
        out["y_label"] = out["y_label_raw"]
        after = out.groupby("split").size().reindex(SPLIT_ORDER, fill_value=0)
        for split in SPLIT_ORDER:
            b = int(before.get(split, 0))
            a = int(after.get(split, 0))
            split_filter_rows.append({
                "split": split,
                "before_count": b,
                "after_count": a,
                "dropped_count": b - a,
                "dropped_fraction": float((b - a) / b) if b else 0.0,
                "filter_rule": f"keep only labels in train-learned top{int(top_k)} vocabulary; no OTHER_ACTIVITY",
            })
    else:
        out["y_label"] = out["y_label_raw"].where(out["y_label_raw"].isin(vocab_set), OTHER_LABEL)
        before = out.groupby("split").size().reindex(SPLIT_ORDER, fill_value=0)
        for split in SPLIT_ORDER:
            b = int(before.get(split, 0))
            split_filter_rows.append({
                "split": split,
                "before_count": b,
                "after_count": b,
                "dropped_count": 0,
                "dropped_fraction": 0.0,
                "filter_rule": "map labels outside train-learned topK vocabulary to OTHER_ACTIVITY",
            })
    out["y_label"] = out["y_label"].astype("string")
    label_meta = {
        "y_mode": y_mode,
        "top_k": int(top_k),
        "topk_vocab_learned_from": "train respondents only",
        "topk_vocab": [str(x) for x in vocab],
        "topk_label_rows": label_rows,
        "closed_world_filtering_by_split": split_filter_rows,
        "other_activity_fraction_all": float(out["y_label"].eq(OTHER_LABEL).mean()) if len(out) else 0.0,
    }
    return out, [str(x) for x in vocab], label_meta


def support_by_split(df: pd.DataFrame, col: str, expected: list[str] | None = None) -> dict[str, Any]:
    out = {}
    levels = expected or sorted(df[col].dropna().astype("string").unique().tolist())
    for split in SPLIT_ORDER:
        sub = df[df["split"].eq(split)]
        counts = sub[col].dropna().astype("string").value_counts().reindex(levels, fill_value=0)
        out[split] = {
            "n_examples": int(len(sub)),
            "n_respondents": int(sub["TUCASEID"].nunique()),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "zero_support": [str(k) for k, v in counts.items() if int(v) == 0],
        }
    return out


def gz_support_by_split(df: pd.DataFrame, g_col: str, z_col: str) -> dict[str, Any]:
    out = {}
    groups = sorted(df[g_col].dropna().astype("string").unique().tolist())
    z_levels = sorted(df[z_col].dropna().astype("string").unique().tolist())
    for split in SPLIT_ORDER:
        sub = df[df["split"].eq(split)]
        counts = sub.groupby([g_col, z_col]).size()
        cells = {}
        vals = []
        for g in groups:
            for z in z_levels:
                value = int(counts.get((g, z), 0))
                cells[f"{g}__{z}"] = value
                vals.append(value)
        out[split] = {"cells": cells, "min_cell_count": int(min(vals)) if vals else 0}
    return out


def feature_source_map(feature_mode: str = "safe") -> dict[str, list[str]]:
    whitelist = feature_whitelist_for(feature_mode)
    return {row["feature"]: list(row["source_columns"]) for row in whitelist["engineered_features"]}


def normalize_where_value(value: Any) -> str:
    if pd.isna(value):
        return "MISSING_WHERE"
    text = str(value).strip()
    if not text:
        return "MISSING_WHERE"
    if text.lower() in {"nan", "none", "na", "n/a", ".", "-1", "-2", "-3"}:
        return "UNKNOWN_WHERE"
    digits = re.sub(r"\D", "", text)
    if digits:
        return f"where_{digits.zfill(2)}"
    return "where_" + slug_label(text)


def build_where_context(examples: pd.DataFrame, top_n: int = 25) -> tuple[pd.Series, pd.DataFrame]:
    source_col = "TEWHERE" if "TEWHERE" in examples.columns else None
    if source_col is None:
        values = pd.Series(["UNAVAILABLE_WHERE"] * len(examples), index=examples.index, dtype="string")
        summary = pd.DataFrame([{
            "source_column": "TEWHERE",
            "raw_value": "UNAVAILABLE",
            "where_context": "UNAVAILABLE_WHERE",
            "train_count": 0,
            "all_count": int(len(examples)),
            "selected_top_category": False,
            "notes": "TEWHERE not present in examples",
        }])
        return values, summary

    normalized = examples[source_col].map(normalize_where_value).astype("string")
    train_norm = normalized[examples["split"].astype("string").eq("train")]
    train_counts = train_norm.value_counts(dropna=False)
    protected = {"MISSING_WHERE", "UNKNOWN_WHERE", "UNAVAILABLE_WHERE"}
    top_values = [v for v in train_counts.index.astype("string").tolist() if v not in protected][: int(top_n)]
    allowed = set(top_values) | protected
    mapped = normalized.where(normalized.isin(allowed), "OTHER_WHERE").astype("string")
    all_counts = mapped.value_counts(dropna=False)
    raw_counts = (
        pd.DataFrame({
            "source_column": source_col,
            "raw_value": examples[source_col].astype("string").fillna("MISSING").astype(str),
            "where_context": normalized.astype(str),
            "split": examples["split"].astype("string"),
        })
        .groupby(["source_column", "raw_value", "where_context", "split"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    train_raw = raw_counts[raw_counts["split"].eq("train")].rename(columns={"count": "train_count"})
    all_raw = (
        raw_counts.groupby(["source_column", "raw_value", "where_context"], dropna=False)["count"]
        .sum()
        .reset_index(name="all_count")
    )
    summary = all_raw.merge(
        train_raw[["source_column", "raw_value", "where_context", "train_count"]],
        on=["source_column", "raw_value", "where_context"],
        how="left",
    )
    summary["train_count"] = summary["train_count"].fillna(0).astype(int)
    summary["mapped_feature_value"] = summary["where_context"].where(summary["where_context"].isin(allowed), "OTHER_WHERE")
    summary["selected_top_category"] = summary["where_context"].isin(top_values)
    summary["notes"] = np.where(
        summary["mapped_feature_value"].eq("OTHER_WHERE"),
        "collapsed to OTHER_WHERE because not in train top categories",
        "kept as explicit strong-context category",
    )
    totals = pd.DataFrame([{
        "source_column": source_col,
        "raw_value": "__FEATURE_TOTAL__",
        "where_context": "__FEATURE_TOTAL__",
        "all_count": int(len(examples)),
        "train_count": int(examples["split"].astype("string").eq("train").sum()),
        "mapped_feature_value": f"{mapped.nunique()} encoded categories",
        "selected_top_category": False,
        "notes": f"top_n_where={top_n}; category vocabulary learned from train split only",
    }])
    summary = pd.concat([totals, summary.sort_values(["all_count", "train_count"], ascending=False)], ignore_index=True)
    return mapped, summary


def source_column_name(source: str) -> str:
    return str(source).split(":", 1)[-1]


def first_available_context_series(examples: pd.DataFrame, source_columns: list[str]) -> tuple[pd.Series | None, str | None, int]:
    series: pd.Series | None = None
    used_source: str | None = None
    available_count = 0
    for source in source_columns:
        col = source_column_name(source)
        if col not in examples.columns:
            continue
        candidate = examples[col]
        non_missing = int(candidate.notna().sum())
        if non_missing == 0:
            available_count += 1
            continue
        available_count += 1
        if series is None:
            series = candidate.copy()
            used_source = source
        else:
            series = series.where(series.notna(), candidate)
            if series.notna().sum() > 0 and used_source is None:
                used_source = source
    return series, used_source, available_count


def normalize_context_category(value: Any, feature: str) -> str:
    if pd.isna(value):
        return f"{feature}__MISSING"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "."}:
        return f"{feature}__MISSING"
    if text in {"-1", "-1.0", "-2", "-2.0", "-3", "-3.0"}:
        return f"{feature}__UNKNOWN"
    digits = re.sub(r"\D", "", text)
    if digits and re.fullmatch(r"-?\d+(\.0)?", text):
        return f"{feature}_{int(float(text)):03d}"
    return f"{feature}__{slug_label(text)}"


def collapse_context_category(series: pd.Series, split: pd.Series, feature: str, top_n: int) -> tuple[pd.Series, dict[str, Any]]:
    normalized = series.map(lambda x: normalize_context_category(x, feature)).astype("string")
    train_values = normalized[split.astype("string").eq("train")]
    protected = {f"{feature}__MISSING", f"{feature}__UNKNOWN"}
    top_values = [
        str(v)
        for v in train_values.value_counts(dropna=False).index.astype("string").tolist()
        if str(v) not in protected
    ][: int(top_n)]
    allowed = set(top_values) | protected
    other_value = f"{feature}__OTHER"
    mapped = normalized.where(normalized.isin(allowed), other_value).astype("string")
    return mapped, {
        "train_top_categories": top_values,
        "encoded_categories": int(mapped.nunique(dropna=False)),
        "other_fraction": float(mapped.eq(other_value).mean()) if len(mapped) else 0.0,
        "top_n": int(top_n),
    }


def clean_numeric_context(series: pd.Series, feature: str) -> tuple[pd.Series, dict[str, Any]]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric < 0)
    if feature in {"ctx_usual_hours_total", "ctx_usual_hours_main", "ctx_spouse_usual_hours"}:
        numeric = numeric.clip(lower=0, upper=120)
    elif feature in {"ctx_weekly_earnings", "ctx_hourly_earnings"}:
        # ATUS/CPS earnings fields commonly have two implied decimals. Scaling is
        # monotone and does not affect HGB splits, but makes summaries readable.
        numeric = (numeric / 100.0).clip(lower=0)
    elif "children" in feature or "child" in feature:
        numeric = numeric.clip(lower=0, upper=20)
    return numeric, {
        "non_missing_fraction": float(numeric.notna().mean()) if len(numeric) else 0.0,
        "median": float(numeric.median()) if numeric.notna().any() else None,
        "p95": float(numeric.quantile(0.95)) if numeric.notna().any() else None,
    }


def build_safe_plus_context(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(index=examples.index)
    summary_rows: list[dict[str, Any]] = []
    split = examples["split"].astype("string") if "split" in examples.columns else pd.Series("train", index=examples.index, dtype="string")
    for spec in SAFE_PLUS_CONTEXT_FEATURES:
        feature = str(spec["feature"])
        source_columns = [str(x) for x in spec["source_columns"]]
        source_series, used_source, available_count = first_available_context_series(examples, source_columns)
        base_row = {
            "feature": feature,
            "type": spec["type"],
            "source_columns": ";".join(source_columns),
            "used_source": used_source or "",
            "source_columns_available": int(available_count),
            "included_in_X": False,
            "encoded_categories": "",
            "missing_fraction": "",
            "other_fraction": "",
            "notes": "",
        }
        if source_series is None:
            base_row["notes"] = "unavailable in respondent/CPS/roster tables loaded for this run"
            summary_rows.append(base_row)
            continue
        if spec["type"] == "numeric":
            cleaned, stats = clean_numeric_context(source_series, feature)
            features[feature] = cleaned
            base_row.update({
                "included_in_X": True,
                "missing_fraction": float(cleaned.isna().mean()) if len(cleaned) else 0.0,
                "notes": f"numeric context; median={stats['median']}; p95={stats['p95']}; missing imputed after feature assembly",
            })
        else:
            mapped, stats = collapse_context_category(source_series, split, feature, int(spec.get("top_n", 25)))
            features[feature] = mapped
            base_row.update({
                "included_in_X": True,
                "encoded_categories": stats["encoded_categories"],
                "missing_fraction": float(mapped.astype("string").str.endswith("__MISSING").mean()) if len(mapped) else 0.0,
                "other_fraction": stats["other_fraction"],
                "notes": f"categorical context; vocabulary learned from train split only; top_n={stats['top_n']}",
            })
        summary_rows.append(base_row)
    return features, pd.DataFrame(summary_rows)


def build_feature_frame(examples: pd.DataFrame, feature_mode: str = "safe") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [
        "start_hour",
        "start_hour_sin",
        "start_hour_cos",
        "start_time_bucket",
        "diary_day",
        "is_weekend",
        "is_holiday",
        "year",
        "age",
        "education",
        "employment_status",
        "marital_status",
        "household_child_context",
        "household_size",
        "region",
    ]
    features = examples[cols].copy()
    where_summary = pd.DataFrame([{
        "source_column": "TEWHERE",
        "raw_value": "__NOT_INCLUDED__",
        "where_context": "__NOT_INCLUDED__",
        "all_count": int(len(examples)),
        "train_count": int(examples["split"].astype("string").eq("train").sum()) if "split" in examples.columns else 0,
        "mapped_feature_value": "__NOT_INCLUDED__",
        "selected_top_category": False,
        "notes": "feature_mode=safe excludes where/location from X",
    }])
    if feature_mode == "strong_where":
        features["where_context"], where_summary = build_where_context(examples)
    context_summary = pd.DataFrame([{
        "feature": "__SAFE_PLUS_CONTEXT_NOT_INCLUDED__",
        "type": "",
        "source_columns": "",
        "used_source": "",
        "source_columns_available": "",
        "included_in_X": False,
        "encoded_categories": "",
        "missing_fraction": "",
        "other_fraction": "",
        "notes": f"feature_mode={feature_mode} uses the base safe whitelist only for respondent/CPS/roster context expansion",
    }])
    if feature_mode == "safe_plus_context":
        context_features, context_summary = build_safe_plus_context(examples)
        features = pd.concat([features, context_features], axis=1)
    elif feature_mode not in {"safe", "strong_where"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    numeric_cols = {
        "start_hour",
        "start_hour_sin",
        "start_hour_cos",
        "year",
        "age",
        "household_size",
        "is_weekend",
        "is_holiday",
        *[str(row["feature"]) for row in SAFE_PLUS_CONTEXT_FEATURES if row["type"] == "numeric"],
    }
    for col in [c for c in features.columns if c in numeric_cols]:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(features[col].median() if pd.notna(features[col].median()) else 0.0)
    for col in [c for c in features.columns if c not in numeric_cols]:
        features[col] = features[col].astype("string").fillna("unknown_or_other")
    return features, where_summary, context_summary


def encoded_feature_shape(features: pd.DataFrame) -> tuple[int, int, list[str]]:
    encoded = pd.get_dummies(features, dummy_na=False)
    return int(encoded.shape[0]), int(encoded.shape[1]), [str(c) for c in encoded.columns]


def leakage_audit(feature_columns: list[str], group: str, z: str, design_name: str | None = None, feature_mode: str = "safe") -> tuple[pd.DataFrame, bool]:
    rows = []
    whitelist = feature_whitelist_for(feature_mode)
    source_lookup = feature_source_map(feature_mode)
    used_base = sorted({col.split("_")[0] if col.startswith("start_time_bucket_") else col for col in feature_columns})
    for feature in [row["feature"] for row in whitelist["engineered_features"]]:
        used = any(c == feature or c.startswith(feature + "_") for c in feature_columns)
        reasons = []
        failed = False
        for pattern, reason in FORBIDDEN_PATTERN_REASONS:
            if feature_mode == "strong_where" and feature == "where_context" and reason == "duration/stop/where/who proxy":
                reasons.append("strong-context ablation allows where/location; excluded from the primary safe feature set")
                continue
            if pattern.search(feature):
                failed = True
                reasons.append(reason)
        rows.append({
            "design": design_name or "",
            "feature": feature,
            "source_columns": ";".join(source_lookup.get(feature, [])),
            "used_in_X": bool(used),
            "selected_group": group,
            "selected_z": z,
            "feature_mode": feature_mode,
            "audit_status": "FAIL" if failed and used else "PASS",
            "notes": "; ".join(reasons) if reasons else "manual whitelist feature",
        })
    for forbidden in whitelist["forbidden_sources"]:
        rows.append({
            "design": design_name or "",
            "feature": f"FORBIDDEN_SOURCE::{forbidden}",
            "source_columns": "",
            "used_in_X": False,
            "selected_group": group,
            "selected_z": z,
            "feature_mode": feature_mode,
            "audit_status": "PASS",
            "notes": "explicitly excluded by whitelist",
        })
    if feature_mode == "strong_where":
        rows.append({
            "design": design_name or "",
            "feature": "STRONG_CONTEXT_SOURCE::where/location",
            "source_columns": "activity:TEWHERE",
            "used_in_X": True,
            "selected_group": group,
            "selected_z": z,
            "feature_mode": feature_mode,
            "audit_status": "PASS",
            "notes": "allowed only for the strong-context efficiency ablation; excluded from primary safe X",
        })
    if feature_mode == "safe_plus_context":
        rows.append({
            "design": design_name or "",
            "feature": "SAFE_PLUS_CONTEXT_SOURCE::respondent_cps_roster_only",
            "source_columns": "respondent;atus_cps;roster",
            "used_in_X": True,
            "selected_group": group,
            "selected_z": z,
            "feature_mode": feature_mode,
            "audit_status": "PASS",
            "notes": "adds only non-episode-proxy respondent/CPS/roster context; where/who/duration/stop/activity-derived variables remain excluded",
        })
    rows.append({
        "design": design_name or "",
        "feature": "PROPENSITY_H_FEATURES",
        "source_columns": "prediction probabilities only",
        "used_in_X": False,
        "selected_group": group,
        "selected_z": z,
        "feature_mode": feature_mode,
        "audit_status": "PASS",
        "notes": "H excludes true G/Y/Z, set size, alpha, threshold, and conformal outputs",
    })
    df = pd.DataFrame(rows)
    pass_flag = not df["audit_status"].eq("FAIL").any()
    return df, bool(pass_flag)


def majority_and_entropy(df: pd.DataFrame, label_col: str) -> dict[str, float]:
    counts = df[label_col].dropna().astype("string").value_counts()
    entropy, effective, majority = entropy_from_counts(counts)
    return {
        "entropy": float(entropy),
        "effective_number_of_labels": float(effective),
        "majority_baseline": float(majority),
    }


def alpha_grid(n_cal: int, n_labels: int) -> list[float]:
    grid = [0.02, 0.05, 0.1, 0.15, 0.2]
    if n_cal < 5000:
        grid = [0.05, 0.1, 0.2]
    if n_labels >= 20:
        grid.append(0.25)
    return sorted(set(grid))


def output_root_for(output_dir: Path, design_name: str) -> Path:
    output_dir = resolve_path(output_dir)
    if output_dir.name == design_name:
        return output_dir.parent
    return output_dir


def write_feature_artifacts(
    root_dir: Path,
    audit: pd.DataFrame,
    audit_pass: bool,
    design_name: str,
    feature_mode: str,
    where_summary: pd.DataFrame,
    context_summary: pd.DataFrame,
) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    active_whitelist = feature_whitelist_for(feature_mode)
    write_json(root_dir / "feature_whitelist.json", active_whitelist)
    write_json(root_dir / "feature_whitelist_safe.json", feature_whitelist_for("safe"))
    if feature_mode == "safe_plus_context":
        write_json(root_dir / "feature_whitelist_safe_plus_context.json", feature_whitelist_for("safe_plus_context"))
    if feature_mode == "strong_where":
        write_json(root_dir / "feature_whitelist_strong_where.json", feature_whitelist_for("strong_where"))
    write_csv(root_dir / "feature_leakage_audit.csv", audit)
    write_csv(root_dir / f"feature_leakage_audit_{feature_mode}.csv", audit)
    write_csv(root_dir / f"feature_leakage_audit_{design_name}.csv", audit)
    write_csv(root_dir / "where_feature_summary.csv", where_summary)
    if feature_mode == "safe_plus_context":
        write_csv(root_dir / "safe_plus_context_added_variables.csv", context_summary)
    lines = [
        "# ATUS Feature Builder Summary",
        "",
        f"- Leakage audit pass: `{audit_pass}`",
        f"- Feature mode: `{feature_mode}`",
        "- Feature construction uses a manual whitelist only; it does not consume `candidate_X=True` from inspection.",
        "- Activity code, tier labels, duration, stop time, who-with, activity summaries, G variables, true Y, and conformal outputs are excluded from X.",
        "- Propensity H is restricted to frozen classifier prediction probabilities/entropy/margin and excludes true G/Y/Z and conformal outputs.",
    "",
        "## Whitelisted Features",
        "",
    ]
    if feature_mode == "safe":
        lines.insert(6, "- Where/location is excluded from safe primary X.")
    elif feature_mode == "strong_where":
        lines.insert(6, "- `where_context` from `TEWHERE` is included only for the strong-context ablation; it can proxy activity and is excluded from the primary safe-X design.")
    else:
        lines.insert(6, "- `safe_plus_context` adds only respondent/CPS/roster context; where/who/duration/stop/activity-derived variables remain excluded.")
    for row in active_whitelist["engineered_features"]:
        lines.append(f"- `{row['feature']}` from `{', '.join(row['source_columns'])}`: {row['allowed_reason']}")
    (root_dir / "feature_builder_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    where_lines = [
        "# ATUS Strong-Where Feature Summary",
        "",
        f"- Feature mode: `{feature_mode}`",
        f"- Included in X: `{feature_mode == 'strong_where'}`",
        "- Status: `strong_context_ablation`",
        "- Source: `activity:TEWHERE`",
        "- Guardrails: activity code, tier labels, duration, stop time, who-with, activity summaries, true G/Y/Z, set size, alpha, threshold, and conformal outputs remain excluded.",
        "- Category vocabulary for where is learned from train split only; non-top categories map to `OTHER_WHERE`; missing/unknown are separate categories.",
        f"- Summary CSV: `{root_dir / 'where_feature_summary.csv'}`",
    ]
    if len(where_summary):
        where_lines.extend(["", "## Encoded Category Overview", ""])
        cols = [c for c in ["mapped_feature_value", "all_count", "train_count", "notes"] if c in where_summary.columns]
        overview = where_summary[cols].drop_duplicates().head(30)
        for _, row in overview.iterrows():
            where_lines.append(
                f"- `{row.get('mapped_feature_value')}`: all={row.get('all_count')}, train={row.get('train_count')} ({row.get('notes')})"
            )
    if feature_mode == "strong_where":
        (root_dir / "strong_where_feature_summary.md").write_text("\n".join(where_lines) + "\n", encoding="utf-8")
    context_lines = [
        "# ATUS Safe Plus Context Feature Summary",
        "",
        f"- Feature mode: `{feature_mode}`",
        f"- Included in X: `{feature_mode == 'safe_plus_context'}`",
        "- Scope: safe X plus non-episode-proxy respondent/CPS/roster context only.",
        "- Explicitly excluded: where/location, who-with, activity code, tier labels, activity descriptions, duration, stop time, activity summary variables, true G/Y/Z, set size, alpha, threshold, and conformal outputs.",
        f"- Added-variable CSV: `{root_dir / 'safe_plus_context_added_variables.csv'}`",
        "",
        "## Added Variables",
        "",
    ]
    included = context_summary[context_summary.get("included_in_X", pd.Series(False, index=context_summary.index)).astype(bool)] if len(context_summary) else pd.DataFrame()
    if len(included):
        for _, row in included.iterrows():
            context_lines.append(
                f"- `{row.get('feature')}` from `{row.get('used_source')}` ({row.get('type')}): {row.get('notes')}"
            )
    else:
        context_lines.append("- None for this feature mode.")
    if feature_mode == "safe_plus_context":
        (root_dir / "safe_plus_context_feature_summary.md").write_text("\n".join(context_lines) + "\n", encoding="utf-8")


def write_prep_aggregate(root_dir: Path, design_name: str, dryrun: dict[str, Any], checks: pd.DataFrame, command: str) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    write_json(root_dir / f"{design_name}_dryrun.json", dryrun)
    existing = []
    checks_path = root_dir / "prep_checks.csv"
    if checks_path.exists():
        existing_df = pd.read_csv(checks_path)
        existing = existing_df[existing_df["design"] != design_name].to_dict("records")
    rows = existing + checks.to_dict("records")
    write_csv(checks_path, rows)
    with (root_dir / "preparation_commands.txt").open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {command}\n")
        f.write(f"result: design={design_name}, dry_run_success={dryrun.get('dry_run_success')}, leakage_audit_pass={dryrun.get('leakage_audit_pass')}, n_train={dryrun.get('split_counts', {}).get('train', {}).get('n_examples')}\n")
    all_checks = pd.read_csv(root_dir / "prep_checks.csv")
    summary_lines = [
        "# ATUS Preparation Summary",
        "",
        "This step validates the design before model fitting.",
        "",
        "| design | dry_run_success | leakage_audit_pass | n_train | n_cal | n_repair | n_test | feature_shape | majority_baseline | alpha_grid |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in all_checks.drop_duplicates("design", keep="last").iterrows():
        summary_lines.append(
            f"| {row['design']} | {row['dry_run_success']} | {row['leakage_audit_pass']} | {row['n_train']} | {row['n_calibration']} | {row['n_repair']} | {row['n_test']} | {row['feature_matrix_shape']} | {row['majority_baseline']} | {row['recommended_alpha_grid']} |"
        )
    (root_dir / "prep_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def prepare_design(
    raw_dir: Path,
    extract_dir: Path,
    output_dir: Path,
    design_name: str,
    unit: str,
    top_k: int,
    group: str,
    z: str,
    seed: int,
    exclude_data_code_activities: bool,
    lexicon_path: Path | None = None,
    use_cache: bool = True,
    feature_mode: str = "safe",
    y_mode: str = "top20_with_other",
) -> dict[str, Any]:
    group_col = {"sex": "sex_g", "race": "race_g"}[group]
    z_col = {"employment": "employment_z", "household": "household_z", "daypart": "start_time_bucket"}[z]
    examples, base_meta, excluded_codes = build_base_examples(
        raw_dir=raw_dir,
        extract_dir=extract_dir,
        seed=seed,
        unit=unit,
        exclude_data_code_activities=exclude_data_code_activities,
        lexicon_path=lexicon_path,
        use_cache=use_cache,
    )
    examples, topk_vocab, label_meta = apply_train_topk_labels(examples, top_k, y_mode=y_mode)
    needed = ["TUCASEID", "split", "y_label", "tier2_code", "tier2_name", group_col, z_col]
    clean = examples.dropna(subset=needed).copy()
    features, where_summary, context_summary = build_feature_frame(clean, feature_mode=feature_mode)
    n_rows, n_cols, encoded_cols = encoded_feature_shape(features)
    audit, audit_pass = leakage_audit(encoded_cols, group=group, z=z, design_name=design_name, feature_mode=feature_mode)
    root_dir = output_root_for(output_dir, design_name)
    write_feature_artifacts(root_dir, audit, audit_pass, design_name, feature_mode, where_summary, context_summary)
    write_csv(root_dir / "excluded_activity_codes.csv", excluded_codes)
    label_map = pd.DataFrame(label_meta.get("topk_label_rows", []))
    if len(label_map):
        label_map = label_map[["tier2_code", "tier2_name", "y_label", "mapping_source", "train_count", "rank"]]
    if y_mode == "top20_with_other":
        other_row = pd.DataFrame([{
            "tier2_code": "OTHER_ACTIVITY",
            "tier2_name": "Other activity outside train topK vocabulary",
            "y_label": OTHER_LABEL,
            "mapping_source": "manual",
            "train_count": int((examples["split"].eq("train") & ~examples["y_label_raw"].isin(set(topk_vocab))).sum()),
            "rank": int(top_k) + 1,
        }])
        label_map_used = pd.concat([label_map, other_row], ignore_index=True)
    else:
        label_map_used = label_map.copy()
    write_csv(root_dir / "activity_label_map_used.csv", label_map_used)
    closed_world = is_closed_world_y_mode(y_mode)
    if closed_world:
        write_csv(root_dir / "closed_world_topk_label_map.csv", label_map_used)
        if y_mode == "closed_world_top20":
            write_csv(root_dir / "closed_world_top20_label_map.csv", label_map_used)

    y_expected = topk_vocab if closed_world else topk_vocab + [OTHER_LABEL]
    y_support = support_by_split(clean, "y_label", y_expected)
    g_support = support_by_split(clean, group_col)
    z_support = support_by_split(clean, z_col)
    gz_support = gz_support_by_split(clean, group_col, z_col)
    if closed_world:
        closed_filter = pd.DataFrame(label_meta.get("closed_world_filtering_by_split", []))
        write_csv(root_dir / "closed_world_topk_filtering_summary.csv", closed_filter)
        if y_mode == "closed_world_top20":
            write_csv(root_dir / "closed_world_top20_filtering_summary.csv", closed_filter)
        support_rows = []
        for split in SPLIT_ORDER:
            support_rows.append({
                "split": split,
                "n_examples": y_support[split]["n_examples"],
                "n_respondents": y_support[split]["n_respondents"],
                "y_label_support": json.dumps(y_support[split]["counts"], sort_keys=True),
                "y_zero_support": json.dumps(y_support[split]["zero_support"], sort_keys=True),
                "g_support": json.dumps(g_support[split]["counts"], sort_keys=True),
                "z_support": json.dumps(z_support[split]["counts"], sort_keys=True),
                "gz_support": json.dumps(gz_support[split]["cells"], sort_keys=True),
                "gz_min_cell_count": gz_support[split]["min_cell_count"],
            })
        write_csv(root_dir / "closed_world_topk_support_by_split.csv", support_rows)
        if y_mode == "closed_world_top20":
            write_csv(root_dir / "closed_world_top20_support_by_split.csv", support_rows)
    split_counts = {
        split: {
            "n_examples": int(len(clean[clean["split"].eq(split)])),
            "n_respondents": int(clean.loc[clean["split"].eq(split), "TUCASEID"].nunique()),
        }
        for split in SPLIT_ORDER
    }
    train_metrics = majority_and_entropy(clean[clean["split"].eq("train")], "y_label")
    cal_n = split_counts["calibration"]["n_examples"]
    alphas = alpha_grid(cal_n, len(y_expected))
    dryrun = {
        "design": design_name,
        "dry_run_success": bool(audit_pass and all(not y_support[s]["zero_support"] for s in SPLIT_ORDER)),
        "unit": unit,
        "seed": seed,
        "y_mode": y_mode,
        "feature_mode": feature_mode,
        "strong_context_ablation": bool(feature_mode == "strong_where"),
        "safe_plus_context": bool(feature_mode == "safe_plus_context"),
        "exclude_data_code_activities": exclude_data_code_activities,
        "label_schema": {
            "tier2_code_column": "tier2_code",
            "tier2_name_column": "tier2_name",
            "y_label_column": "y_label",
            **label_meta,
        },
        "lexicon_mapping_status": base_meta.get("lexicon_mapping_status"),
        "lexicon_notes": base_meta.get("notes"),
        "activity_label_map_used_path": str(root_dir / "activity_label_map_used.csv"),
        "fallback_or_manual_y_labels": sorted(
            label_map_used.loc[label_map_used["mapping_source"].isin(["fallback", "manual"]), "y_label"].astype("string").unique().tolist()
        ),
        "data_code_activity_handling": {
            "rule": base_meta.get("data_code_activity_rule"),
            "excluded": bool(exclude_data_code_activities),
            "rows_total": base_meta.get("data_code_activity_rows_total"),
            "rows_after_sleep_filter": base_meta.get("data_code_activity_rows_after_sleep_filter"),
            "excluded_activity_codes_csv": str(root_dir / "excluded_activity_codes.csv"),
        },
        "group": {"name": group, "column": group_col},
        "z": {"name": z, "column": z_col},
        "split_counts": split_counts,
        "y_support_by_split": y_support,
        "g_support_by_split": g_support,
        "z_support_by_split": z_support,
        "gz_support_by_split": gz_support,
        "feature_matrix_shape": [n_rows, n_cols],
        "feature_columns_preview": encoded_cols[:30],
        "feature_whitelist_path": str(root_dir / "feature_whitelist.json"),
        "feature_whitelist_safe_path": str(root_dir / "feature_whitelist_safe.json"),
        "feature_whitelist_safe_plus_context_path": str(root_dir / "feature_whitelist_safe_plus_context.json") if feature_mode == "safe_plus_context" else None,
        "feature_whitelist_strong_where_path": str(root_dir / "feature_whitelist_strong_where.json") if feature_mode == "strong_where" else None,
        "feature_leakage_audit_path": str(root_dir / "feature_leakage_audit.csv"),
        "feature_leakage_audit_mode_path": str(root_dir / f"feature_leakage_audit_{feature_mode}.csv"),
        "feature_whitelist_closed_world_safe_path": str(root_dir / "feature_whitelist_closed_world_safe.json") if closed_world and feature_mode == "safe" else None,
        "feature_leakage_audit_closed_world_top20_path": str(root_dir / "feature_leakage_audit_closed_world_top20.csv") if y_mode == "closed_world_top20" and feature_mode == "safe" else None,
        "feature_leakage_audit_closed_world_topk_path": str(root_dir / "feature_leakage_audit_closed_world_topk.csv") if closed_world and feature_mode == "safe" else None,
        "where_feature_summary_path": str(root_dir / "where_feature_summary.csv"),
        "strong_where_feature_summary_path": str(root_dir / "strong_where_feature_summary.md") if feature_mode == "strong_where" else None,
        "safe_plus_context_feature_summary_path": str(root_dir / "safe_plus_context_feature_summary.md") if feature_mode == "safe_plus_context" else None,
        "safe_plus_context_added_variables_path": str(root_dir / "safe_plus_context_added_variables.csv") if feature_mode == "safe_plus_context" else None,
        "safe_plus_context_added_variables_count": int(
            context_summary["included_in_X"].astype(bool).sum()
        ) if "included_in_X" in context_summary.columns else 0,
        "leakage_audit_pass": bool(audit_pass),
        "majority_baseline": train_metrics["majority_baseline"],
        "effective_number_of_labels": train_metrics["effective_number_of_labels"],
        "entropy": train_metrics["entropy"],
        "recommended_alpha_grid": alphas,
        "output_paths": {
            "dryrun_json": str(root_dir / f"{design_name}_dryrun.json"),
            "prep_checks_csv": str(root_dir / "prep_checks.csv"),
            "prep_summary_md": str(root_dir / "prep_summary.md"),
        },
        "read_metadata": base_meta,
    }
    if closed_world and feature_mode == "safe":
        closed_whitelist = feature_whitelist_for("safe")
        closed_whitelist["y_mode"] = y_mode
        closed_whitelist["y_top_k"] = int(top_k)
        closed_whitelist["closed_world_label_rule"] = f"top{int(top_k)} vocabulary is learned from train respondents only; non-topK examples are dropped in every split; no OTHER_ACTIVITY label is created"
        write_json(root_dir / "feature_whitelist_closed_world_safe.json", closed_whitelist)
        write_csv(root_dir / "feature_leakage_audit_closed_world_topk.csv", audit)
        if y_mode == "closed_world_topk" and int(top_k) == 10:
            closed_whitelist["selected_design"] = "closed_world_top10_safe"
            closed_whitelist["primary_design_candidate"] = True
            write_json(root_dir / "feature_whitelist_selected_top10_safe.json", closed_whitelist)
            write_csv(root_dir / "feature_leakage_audit_selected_top10.csv", audit)
        if y_mode == "closed_world_top20":
            write_csv(root_dir / "feature_leakage_audit_closed_world_top20.csv", audit)
    checks = pd.DataFrame([{
        "design": design_name,
        "y_mode": y_mode,
        "feature_mode": feature_mode,
        "dry_run_success": dryrun["dry_run_success"],
        "leakage_audit_pass": bool(audit_pass),
        "n_train": split_counts["train"]["n_examples"],
        "n_calibration": split_counts["calibration"]["n_examples"],
        "n_repair": split_counts["repair"]["n_examples"],
        "n_test": split_counts["test"]["n_examples"],
        "feature_matrix_shape": f"{n_rows}x{n_cols}",
        "majority_baseline": dryrun["majority_baseline"],
        "effective_number_of_labels": dryrun["effective_number_of_labels"],
        "recommended_alpha_grid": json.dumps(alphas),
        "y_zero_support_any": any(bool(y_support[s]["zero_support"]) for s in SPLIT_ORDER),
        "min_gz_cell": min(gz_support[s]["min_cell_count"] for s in SPLIT_ORDER),
        "lexicon_mapping_status": dryrun["lexicon_mapping_status"],
        "data_code_rows_after_sleep_filter": base_meta.get("data_code_activity_rows_after_sleep_filter"),
        "safe_plus_context_added_variables_count": dryrun["safe_plus_context_added_variables_count"],
    }])
    return {
        "examples": clean,
        "features": features,
        "dryrun": dryrun,
        "checks": checks,
        "audit": audit,
        "root_dir": root_dir,
    }


def json_dumpable(value: Any) -> Any:
    return json_safe(value)
