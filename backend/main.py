from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from io import BytesIO
from typing import Dict, List
import re
import math
import urllib.request
import urllib.error
import json

app = FastAPI(title="Sustainment Prediction Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_file(file: UploadFile) -> pd.DataFrame:
    contents = file.file.read()
    file.file.seek(0)
    name_lower = (file.filename or "").lower()

    def parse_csv_bytes(raw: bytes) -> pd.DataFrame:
        encodings = ["utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16le", "utf-16be"]
        last_error = None
        for enc in encodings:
            try:
                return pd.read_csv(
                    BytesIO(raw),
                    encoding=enc,
                    on_bad_lines="skip",
                    engine="python",
                )
            except Exception as e:
                last_error = e
                continue

        try:
            return pd.read_csv(
                BytesIO(raw),
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="skip",
                engine="python",
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error parsing file after multiple fallbacks: {str(e)}",
            )

    try:
        looks_like_csv = name_lower.endswith(".csv") or b"," in contents[:1024] or b"\n" in contents[:1024]

        if looks_like_csv:
            try:
                df = parse_csv_bytes(contents)
            except Exception:
                try:
                    df = pd.read_excel(BytesIO(contents))
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Error parsing file (csv/xlsx attempt): {str(exc)}")
        else:
            try:
                df = pd.read_excel(BytesIO(contents))
            except Exception:
                df = parse_csv_bytes(contents)

        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")


def extract_facilities_and_systems(inventory_df: pd.DataFrame) -> Dict[str, Dict]:
    facilities_dict = {}
    df_cleaned = inventory_df.dropna(how="all").dropna(axis=1, how="all")
    
    facility_col = None
    systems_col = None
    occupancy_col = None
    life_exp_col = None
    header_row_idx = None
    
    for idx, row in df_cleaned.iterrows():
        row_lower = [str(v).strip().lower() if pd.notna(v) else "" for v in row.tolist()]
        if "facilities" in " ".join(row_lower) and "systems" in " ".join(row_lower):
            header_row_idx = idx
            for j, val in enumerate(row_lower):
                if "facilities" in val and facility_col is None:
                    facility_col = j
                if "systems" in val and systems_col is None:
                    systems_col = j
                if ("life expectancy" in val or val.replace(" ", "") == "lifeexpectancy") and life_exp_col is None:
                    life_exp_col = j
            break
    
    if header_row_idx is not None:
        for j, col in enumerate(df_cleaned.columns):
            col_lower = str(col).lower()
            if "occup" in col_lower or "staff" in col_lower:
                occupancy_col = j
                break
    
    if header_row_idx is not None:
        for ridx, row in df_cleaned.iterrows():
            if ridx <= header_row_idx:
                continue
            
            facility_name = str(row.iloc[facility_col]).strip() if facility_col is not None and facility_col < len(row) else ""
            systems_text = str(row.iloc[systems_col]).strip() if systems_col is not None and systems_col < len(row) else ""
            occupancy = str(row.iloc[occupancy_col]).strip() if occupancy_col is not None and occupancy_col < len(row) else ""
            
            if not facility_name or facility_name.lower() in {"facilities", "facility", "nan", ""}:
                continue
            
            systems_list = []
            if systems_text and systems_text.lower() not in {"nan", ""}:
                if "all" in systems_text.lower():
                    systems_list = ["all"]
                else:
                    systems_list = [s.strip() for s in systems_text.split(",") if s.strip()]
            
            facilities_dict[facility_name] = {
                "systems": systems_list,
                "occupancy": occupancy.lower() if occupancy else "unknown",
                "name": facility_name
            }
    
    return facilities_dict


def extract_systems_and_life_expectancy(inventory_df: pd.DataFrame) -> Dict[str, int]:
    systems_dict = {}

    df_cleaned = inventory_df.dropna(how="all").dropna(axis=1, how="all")

    header_system_col = None
    header_life_col = None
    header_row_idx = None

    for idx, row in df_cleaned.iterrows():
        row_lower = [str(v).strip().lower() if pd.notna(v) else "" for v in row.tolist()]
        for j, val in enumerate(row_lower):
            if "system" in val and header_system_col is None:
                header_system_col = j
                header_row_idx = idx
            if ("life expectancy" in val or val.replace(" ", "") == "lifeexpectancy") and header_life_col is None:
                header_life_col = j
                header_row_idx = idx if header_row_idx is None else header_row_idx
        if header_system_col is not None and header_life_col is not None:
            break

    if header_system_col is not None and header_life_col is not None:
        for ridx, row in df_cleaned.iterrows():
            if header_row_idx is not None and ridx <= header_row_idx:
                continue
            system_raw = row.iloc[header_system_col] if header_system_col < len(row) else None
            life_raw = row.iloc[header_life_col] if header_life_col < len(row) else None
            if pd.isna(system_raw) or pd.isna(life_raw):
                continue
            system = str(system_raw).strip()
            if not system or system.lower() in {"system", "systems"}:
                continue
            try:
                life_int = int(float(life_raw))
            except (TypeError, ValueError):
                continue
            if life_int > 0:
                systems_dict[system] = life_int

    if not systems_dict:
        systems_col = None
        life_exp_col = None

        for col in df_cleaned.columns:
            col_lower = str(col).lower()
            if "system" in col_lower and systems_col is None:
                systems_col = col
            if "life expectancy" in col_lower or "lifeexpectancy" in col_lower.replace(" ", ""):
                life_exp_col = col

        if systems_col is None:
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == "object":
                    sample_values = df_cleaned[col].dropna().head(10).astype(str).tolist()
                    if any(
                        val.lower() in ["foundation", "basement", "superstructure", "hvac", "electric", "plumbing"]
                        for val in sample_values
                    ):
                        systems_col = col
                        break

        if life_exp_col is None:
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype in ["int64", "float64"]:
                    values = df_cleaned[col].dropna()
                    if len(values) > 0 and 10 <= values.min() <= 100 and values.max() <= 100:
                        life_exp_col = col
                        break

        if systems_col and life_exp_col:
            for _, row in df_cleaned.iterrows():
                system = str(row[systems_col]).strip()
                life_exp = row[life_exp_col]

                if pd.notna(system) and pd.notna(life_exp) and system.lower() not in ["system", "systems", "nan", ""]:
                    try:
                        life_exp_int = int(float(life_exp))
                        if life_exp_int > 0:
                            systems_dict[system] = life_exp_int
                    except (ValueError, TypeError):
                        continue

    return systems_dict


def extract_projects(projects_df: pd.DataFrame) -> Dict[int, List[Dict]]:
    projects_by_year: Dict[int, List[Dict]] = {}
    df_cleaned = projects_df.dropna(how="all").dropna(axis=1, how="all")

    fiscal_year_col = None
    cost_col = None
    scope_col = None
    title_col = None
    site_col = None
    building_col = None

    def extract_year_from_scope(scope_text: str) -> int | None:
        if not scope_text:
            return None
        text = scope_text.lower()
        patterns = [r"start\s*year[:\s]*([12][0-9]{3})", r"fiscal\s*year[:\s]*([12][0-9]{3})"]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    year = int(m.group(1))
                    if 2000 <= year <= 2100:
                        return year
                except Exception:
                    continue
        for token in re.findall(r"([12][0-9]{3})", text):
            year = int(token)
            if 2000 <= year <= 2100:
                return year
        return None

    def extract_cost_from_scope(scope_text: str) -> str:
        if not scope_text:
            return ""
        m = re.search(r"[Cc]ost[:\s]*\$?([0-9.,]+)\s*([mMkK]?)", scope_text)
        if m:
            amount = m.group(1)
            suffix = m.group(2).upper()
            return f"${amount}{suffix}"
        return ""

    def try_parse_year(value) -> int | None:
        if pd.isna(value):
            return None
        try:
            num = int(float(value))
            if 1900 <= num <= 2100:
                return num
        except (ValueError, TypeError):
            pass
        try:
            text = str(value)
            for token in text.replace("/", " ").replace("-", " ").split():
                if token.isdigit() and len(token) == 4:
                    num = int(token)
                    if 1900 <= num <= 2100:
                        return num
        except Exception:
            return None
        return None

    for col in df_cleaned.columns:
        col_lower = str(col).lower()
        if "fiscal year" in col_lower or col_lower in {"fy", "fyear"} or "year" == col_lower:
            fiscal_year_col = col
        if "cost" in col_lower or "budget" in col_lower or "amount" in col_lower:
            cost_col = col
        if "scope" in col_lower or "description" in col_lower:
            scope_col = col
        if "project" in col_lower and "title" in col_lower:
            title_col = col
        if "site" in col_lower:
            site_col = col
        if "building" in col_lower and "number" in col_lower:
            building_col = col

    if fiscal_year_col is None:
        for col in df_cleaned.columns:
            sample_values = df_cleaned[col].dropna().head(50)
            years = [try_parse_year(v) for v in sample_values]
            years = [y for y in years if y]
            if years:
                fiscal_year_col = col
                break

    if not fiscal_year_col:
        return projects_by_year

    for _, row in df_cleaned.iterrows():
        fiscal_year = try_parse_year(row[fiscal_year_col]) if fiscal_year_col else None

        scope_text = str(row[scope_col]) if scope_col and pd.notna(row.get(scope_col, None)) else ""
        if fiscal_year is None:
            fiscal_year = extract_year_from_scope(scope_text)

        if fiscal_year is None:
            continue

        if 2000 <= fiscal_year <= 2100:
            if fiscal_year not in projects_by_year:
                projects_by_year[fiscal_year] = []

            title_text = str(row[title_col]) if title_col and pd.notna(row.get(title_col, None)) else ""
            cost_val = float(row[cost_col]) if cost_col and pd.notna(row.get(cost_col, None)) else 0
            cost_text = extract_cost_from_scope(scope_text)
            site_text = str(row[site_col]) if site_col and pd.notna(row.get(site_col, None)) else ""
            building_text = str(row[building_col]) if building_col and pd.notna(row.get(building_col, None)) else ""

            project_data = {
                "cost": cost_val,
                "cost_text": cost_text,
                "title": title_text,
                "site": site_text,
                "building": building_text,
                "scope": scope_text,
            }
            projects_by_year[fiscal_year].append(project_data)
        else:
            inferred_year = extract_year_from_scope(scope_text)
            if inferred_year:
                if inferred_year not in projects_by_year:
                    projects_by_year[inferred_year] = []
                title_text = str(row[title_col]) if title_col and pd.notna(row.get(title_col, None)) else ""
                cost_text = extract_cost_from_scope(scope_text)
                site_text = str(row[site_col]) if site_col and pd.notna(row.get(site_col, None)) else ""
                building_text = str(row[building_col]) if building_col and pd.notna(row.get(building_col, None)) else ""
                project_data = {
                    "cost": float(row[cost_col]) if cost_col and pd.notna(row.get(cost_col, None)) else 0,
                    "cost_text": cost_text,
                    "title": title_text,
                    "site": site_text,
                    "building": building_text,
                    "scope": scope_text,
                }
                projects_by_year[inferred_year].append(project_data)

    return projects_by_year


def identify_systems_from_scope(scope_text: str, available_systems: List[str]) -> List[str]:
    scope_lower = scope_text.lower()
    identified_systems = []

    system_keywords = {
        "hvac": [
            "hvac",
            "heating",
            "cooling",
            "air handling",
            "chiller",
            "compressor",
            "ductwork",
            "airflow",
            "temperature",
            "refrigerant",
            "cooling tower",
            "air conditioning",
        ],
        "electric": [
            "electrical",
            "electric",
            "power",
            "wiring",
            "panel",
            "circuit",
            "transformer",
            "substation",
            "switchgear",
            "generator",
            "ups",
            "distribution",
        ],
        "plumbing": ["plumbing", "water", "piping", "pump", "water treatment"],
        "roofing": ["roof", "roofing", "roof penetration"],
        "foundation": ["foundation"],
        "superstructure": ["superstructure", "structure"],
        "exterior structure": ["exterior", "exterior structure"],
        "interior construction": ["interior construction", "interior"],
        "interior finishes": ["interior finishes", "finishes"],
        "fire protection": ["fire protection", "fire", "sprinkler"],
        "utilities": ["utility", "utilities"],
        "distribution": ["distribution", "delivery system"],
        "special construction": ["special construction"],
        "improvements": ["improvements"],
        "protection": ["protection"],
        "conveying": ["conveying", "elevator"],
        "stairs": ["stairs", "stair"],
        "furnishing": ["furnishing", "furniture"],
    }

    for system in available_systems:
        system_lower = system.lower()
        keywords = system_keywords.get(system_lower, [system_lower])

        if any(keyword in scope_lower for keyword in keywords):
            identified_systems.append(system)

    return identified_systems


def calculate_degradation_rate(life_expectancy: int) -> float:
    return 100.0 / life_expectancy if life_expectancy > 0 else 0


def calculate_system_importance(systems: Dict[str, int], facilities: Dict[str, Dict], projects_by_year: Dict[int, List[Dict]]) -> Dict[str, float]:
    importance = {system: 0.0 for system in systems.keys()}
    
    facility_counts = {system: 0 for system in systems.keys()}
    for facility_data in facilities.values():
        facility_systems = facility_data.get("systems", [])
        if "all" in [s.lower() for s in facility_systems]:
            for system in systems.keys():
                facility_counts[system] += 1
        else:
            for system in systems.keys():
                if any(system.lower() in str(s).lower() for s in facility_systems):
                    facility_counts[system] += 1
    
    project_counts = {system: 0 for system in systems.keys()}
    total_costs = {system: 0.0 for system in systems.keys()}
    
    for year, projects in projects_by_year.items():
        for project in projects:
            scope = project.get("scope", "")
            cost = project.get("cost", 0.0)
            affected_systems = identify_systems_from_scope(scope, list(systems.keys()))
            
            if not affected_systems:
                affected_systems = list(systems.keys())
            
            for system in affected_systems:
                project_counts[system] += 1
                total_costs[system] += cost
    
    max_facilities = max(facility_counts.values()) if facility_counts.values() else 1
    max_projects = max(project_counts.values()) if project_counts.values() else 1
    max_cost = max(total_costs.values()) if total_costs.values() else 1
    
    for system in systems.keys():
        facility_score = facility_counts[system] / max_facilities if max_facilities > 0 else 0
        project_score = project_counts[system] / max_projects if max_projects > 0 else 0
        cost_score = total_costs[system] / max_cost if max_cost > 0 else 0
        
        importance[system] = (facility_score * 0.4 + project_score * 0.3 + cost_score * 0.3) * 100
    
    return importance


def calculate_system_correlations(systems: Dict[str, int], facilities: Dict[str, Dict], projects_by_year: Dict[int, List[Dict]]) -> Dict[str, List[tuple]]:
    correlations = {system: [] for system in systems.keys()}
    
    cooccurrence = {sys1: {sys2: 0 for sys2 in systems.keys()} for sys1 in systems.keys()}
    
    for facility_data in facilities.values():
        facility_systems = facility_data.get("systems", [])
        if "all" in [s.lower() for s in facility_systems]:
            for sys1 in systems.keys():
                for sys2 in systems.keys():
                    if sys1 != sys2:
                        cooccurrence[sys1][sys2] += 1
        else:
            facility_system_names = [s for s in facility_systems if str(s).strip()]
            for sys1 in systems.keys():
                if any(sys1.lower() in str(s).lower() for s in facility_system_names):
                    for sys2 in systems.keys():
                        if sys1 != sys2 and any(sys2.lower() in str(s).lower() for s in facility_system_names):
                            cooccurrence[sys1][sys2] += 1
    
    for year, projects in projects_by_year.items():
        for project in projects:
            scope = project.get("scope", "")
            affected_systems = identify_systems_from_scope(scope, list(systems.keys()))
            
            if not affected_systems:
                affected_systems = list(systems.keys())
            
            for sys1 in affected_systems:
                for sys2 in affected_systems:
                    if sys1 != sys2:
                        cooccurrence[sys1][sys2] += 1
    
    for sys1 in systems.keys():
        max_cooccur = max(cooccurrence[sys1].values()) if cooccurrence[sys1].values() else 1
        for sys2 in systems.keys():
            if sys1 != sys2 and max_cooccur > 0:
                score = (cooccurrence[sys1][sys2] / max_cooccur) * 100
                if score > 10:
                    correlations[sys1].append((sys2, round(score, 1)))
        
        correlations[sys1].sort(key=lambda x: x[1], reverse=True)
        correlations[sys1] = correlations[sys1][:5]
    
    return correlations


def predict_failure_dates(systems: Dict[str, int], current_fci: Dict[str, float], degradation_rates: Dict[str, float], start_year: int) -> Dict[str, Dict]:
    predictions = {}
    
    for system, life_exp in systems.items():
        current_condition = current_fci.get(system, 100.0)
        deg_rate = degradation_rates.get(system, 0.0)
        
        if deg_rate <= 0:
            predictions[system] = {
                "failure_year": None,
                "remaining_years": None,
                "current_fci": current_condition
            }
            continue
        
        years_to_failure = current_condition / deg_rate if deg_rate > 0 else None
        failure_year = int(start_year + years_to_failure) if years_to_failure else None
        
        predictions[system] = {
            "failure_year": failure_year,
            "remaining_years": round(years_to_failure, 1) if years_to_failure else None,
            "current_fci": round(current_condition, 1),
            "degradation_rate": round(deg_rate, 2)
        }
    
    return predictions


def generate_system_description_from_data(system_name: str, importance: float, correlations: List[tuple], failure_pred: Dict, facilities_count: int) -> Dict[str, str]:
    importance_level = "critical" if importance > 70 else "high" if importance > 40 else "moderate" if importance > 20 else "low"
    
    if failure_pred.get("remaining_years"):
        if failure_pred["remaining_years"] > 30:
            good_desc = f"{system_name} is in excellent condition with {failure_pred['remaining_years']:.0f} years remaining. Fully operational across {facilities_count} facilities."
            fair_desc = f"{system_name} shows moderate wear with {failure_pred['remaining_years']:.0f} years remaining. Functional but requires monitoring across {facilities_count} facilities."
            poor_desc = f"{system_name} has significant degradation with only {failure_pred['remaining_years']:.0f} years remaining. Operational risk affecting {facilities_count} facilities."
        elif failure_pred["remaining_years"] > 10:
            good_desc = f"{system_name} is functional with {failure_pred['remaining_years']:.0f} years remaining. Critical for {facilities_count} facilities."
            fair_desc = f"{system_name} shows wear with {failure_pred['remaining_years']:.0f} years remaining. Needs attention across {facilities_count} facilities."
            poor_desc = f"{system_name} is near failure with only {failure_pred['remaining_years']:.0f} years remaining. High risk to {facilities_count} facilities."
        else:
            good_desc = f"{system_name} is near end of life with {failure_pred['remaining_years']:.0f} years remaining. Critical for {facilities_count} facilities."
            fair_desc = f"{system_name} is failing with {failure_pred['remaining_years']:.0f} years remaining. Urgent attention needed for {facilities_count} facilities."
            poor_desc = f"{system_name} is at failure point with {failure_pred['remaining_years']:.0f} years remaining. Immediate replacement required for {facilities_count} facilities."
    else:
        good_desc = f"{system_name} is operational. Used by {facilities_count} facilities."
        fair_desc = f"{system_name} requires maintenance. Affects {facilities_count} facilities."
        poor_desc = f"{system_name} is at risk. Critical for {facilities_count} facilities."
    
    corr_systems = [s[0] for s in correlations[:3]]
    if corr_systems:
        impact_desc = f"{system_name} failure would affect {len(corr_systems)} related systems ({', '.join(corr_systems[:2])}{' and more' if len(corr_systems) > 2 else ''}) across {facilities_count} facilities. Importance: {importance_level} ({importance:.0f}%)."
    else:
        impact_desc = f"{system_name} is used by {facilities_count} facilities. Importance: {importance_level} ({importance:.0f}%)."
    
    return {
        "good": good_desc,
        "fair": fair_desc,
        "poor": poor_desc,
        "impact": impact_desc,
        "importance": importance,
        "failure_year": failure_pred.get("failure_year"),
        "remaining_years": failure_pred.get("remaining_years")
    }


def get_system_description_legacy(system_name: str) -> Dict[str, str]:
    system_lower = system_name.lower()
    
    descriptions = {
        "foundation": {
            "good": "Structurally sound, no settlement or cracks. Supports building loads without issues.",
            "fair": "Minor settlement or hairline cracks. Requires monitoring but still functional.",
            "poor": "Significant settlement, visible cracks, or structural movement. Risk of building instability.",
            "impact": "Foundation failure can cause structural damage to entire building, affecting all systems above.",
            "dependencies": "Affects: Superstructure, Exterior Structure, Interior Construction, all building systems"
        },
        "basement": {
            "good": "Waterproof, no leaks, proper drainage. Fully usable space.",
            "fair": "Occasional minor leaks or moisture. Some areas may need attention.",
            "poor": "Persistent leaks, water intrusion, or structural issues. Unusable or hazardous.",
            "impact": "Basement failure can damage utilities, cause mold, and compromise building structure.",
            "dependencies": "Affects: Plumbing, Electric, HVAC, Foundation (if water damage occurs)"
        },
        "superstructure": {
            "good": "Load-bearing elements intact. No structural concerns. Meets all safety standards.",
            "fair": "Some wear or minor issues. May need repairs but still safe and functional.",
            "poor": "Significant structural degradation. Safety concerns. May require evacuation.",
            "impact": "Superstructure failure can cause building collapse, affecting all occupants and systems.",
            "dependencies": "Affects: All interior systems, Roofing, Exterior Structure, occupant safety"
        },
        "exterior structure": {
            "good": "Weather-tight, no water intrusion. Proper insulation and protection.",
            "fair": "Minor leaks or wear. Some areas need maintenance but functional.",
            "poor": "Water intrusion, deterioration, or failure. Interior damage likely.",
            "impact": "Exterior failure allows weather intrusion, damaging interior systems and finishes.",
            "dependencies": "Affects: Interior Finishes, HVAC (moisture control), Interior Construction"
        },
        "roofing": {
            "good": "Waterproof, no leaks. Proper drainage and insulation. Expected lifespan remaining.",
            "fair": "Minor leaks or wear. Some repairs needed but functional.",
            "poor": "Persistent leaks, significant deterioration, or failure. Interior damage occurring.",
            "impact": "Roof failure causes water damage to all systems below, including structure and interiors.",
            "dependencies": "Affects: Interior Finishes, Interior Construction, HVAC, Electric (if leaks reach)"
        },
        "hvac": {
            "good": "Full capacity operation. Efficient climate control. All zones functional.",
            "fair": "Reduced capacity or efficiency. Some zones may have issues. Higher energy costs.",
            "poor": "System failure or severe degradation. Inadequate climate control. Occupant discomfort or safety risk.",
            "impact": "HVAC failure affects occupant comfort, equipment operation, and can cause mold/moisture issues.",
            "dependencies": "Affects: Electric (power consumption), Interior Finishes (moisture), occupant operations"
        },
        "electric": {
            "good": "Full capacity, reliable power. All circuits functional. Meets code requirements.",
            "fair": "Some circuits unreliable or overloaded. May need upgrades. Occasional outages.",
            "poor": "Frequent outages, safety hazards, or insufficient capacity. Critical systems at risk.",
            "impact": "Electric failure shuts down all powered systems: HVAC, lighting, equipment, security.",
            "dependencies": "Affects: All powered systems (HVAC, Fire Protection, Conveying, etc.)"
        },
        "plumbing": {
            "good": "No leaks, proper pressure, all fixtures functional. Water quality meets standards.",
            "fair": "Minor leaks or pressure issues. Some fixtures need repair but mostly functional.",
            "poor": "Major leaks, low pressure, or contamination. Water damage or health hazards.",
            "impact": "Plumbing failure causes water damage, health risks, and building closure.",
            "dependencies": "Affects: Interior Finishes, Basement (if leaks), Fire Protection (sprinklers)"
        },
        "fire protection": {
            "good": "All systems operational. Sprinklers, alarms, and suppression ready. Meets code.",
            "fair": "Some systems need maintenance. Minor issues but still functional.",
            "poor": "System failures or code violations. Life safety risk. Building may be unoccupiable.",
            "impact": "Fire protection failure creates life safety hazard and may violate occupancy permits.",
            "dependencies": "Affects: Electric (power for alarms), Plumbing (sprinkler water supply), building occupancy"
        },
    }
    
    for key, desc in descriptions.items():
        if key in system_lower:
            return desc
    
    return {
        "good": f"{system_name} is in excellent condition, fully functional, and meets all operational requirements.",
        "fair": f"{system_name} shows signs of wear but remains functional. Some maintenance or repairs needed.",
        "poor": f"{system_name} has significant degradation or failures. Operational risk or failure likely.",
        "impact": f"{system_name} failure affects facility operations and may impact related systems.",
        "dependencies": "May affect other building systems depending on component type."
    }


def get_system_dependencies(system_name: str, all_systems: List[str]) -> List[str]:
    system_lower = system_name.lower()
    dependencies = []
    
    if "foundation" in system_lower:
        for sys in all_systems:
            sys_lower = sys.lower()
            if any(x in sys_lower for x in ["superstructure", "exterior", "interior", "roofing", "basement"]):
                dependencies.append(sys)
    
    if any(x in system_lower for x in ["superstructure", "exterior structure"]):
        for sys in all_systems:
            sys_lower = sys.lower()
            if any(x in sys_lower for x in ["interior", "roofing", "exterior"]):
                if sys != system_name:
                    dependencies.append(sys)
    
    if "roofing" in system_lower:
        for sys in all_systems:
            sys_lower = sys.lower()
            if any(x in sys_lower for x in ["interior", "hvac", "electric"]):
                dependencies.append(sys)
    
    if "electric" in system_lower:
        for sys in all_systems:
            sys_lower = sys.lower()
            if any(x in sys_lower for x in ["hvac", "fire", "conveying", "plumbing"]):
                dependencies.append(sys)
    
    if "hvac" in system_lower:
        for sys in all_systems:
            sys_lower = sys.lower()
            if "interior" in sys_lower or "finish" in sys_lower:
                dependencies.append(sys)
    
    if "plumbing" in system_lower:
        for sys in all_systems:
            sys_lower = sys.lower()
            if "fire" in sys_lower or "interior" in sys_lower:
                dependencies.append(sys)
    
    return list(set(dependencies))


def calculate_markov_transition_matrix(life_expectancy: int) -> Dict:
    degradation_rate = calculate_degradation_rate(life_expectancy)
    
    p_degrade = min(degradation_rate / 33.0, 1.0)
    p_stay = 1.0 - p_degrade
    
    p_improve = min(0.1, p_degrade * 0.3)
    
    transition_matrix = {
        "states": ["Good", "Fair", "Poor"],
        "matrix": [
            [round(p_stay, 3), round(p_degrade, 3), 0.0],
            [round(p_improve, 3), round(p_stay - p_improve, 3), round(p_degrade, 3)],
            [0.0, round(p_improve, 3), round(1.0 - p_improve, 3)],
        ],
        "degradation_rate_per_year": round(degradation_rate, 2),
        "years_to_failure": life_expectancy,
    }
    
    return transition_matrix


def render_plantuml_svg(plantuml_text: str) -> str | None:
    try:
        url = "https://kroki.io/plantuml/svg"
        payload = {"diagram_source": plantuml_text}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status == 200:
                return response.read().decode("utf-8")
    except Exception as e:
        print(f"Kroki render error: {e}")
        print(f"Diagram start: {plantuml_text[:500]}")
        return None
    return None


def build_markov_plantuml(
    systems: Dict[str, int],
    markov_matrices: Dict[str, Dict],
    projects_by_year: Dict[int, List[Dict]],
    facilities: Dict[str, Dict] | None = None,
    system_descriptions: Dict[str, Dict] | None = None,
    failure_predictions: Dict[str, Dict] | None = None,
) -> str:
    def slug(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", name or "sys")
        cleaned = cleaned[:40] if cleaned else "Sys"
        if cleaned and cleaned[0].isdigit():
            return f"ID_{cleaned}"
        return cleaned
    
    def escape_label(text: str) -> str:
        if not text:
            return "Unknown"
        text = text.replace('"', "'").replace('\n', '\\n').replace(':', '-')
        text = text.replace('<', '[').replace('>', ']')
        text = text.replace('(', '[').replace(')', ']')
        return text[:100]

    lines: List[str] = []
    lines.append("@startuml")
    lines.append("scale 1200 width")
    lines.append("skinparam backgroundColor #0f172a")
    lines.append("skinparam shadowing false")
    lines.append("skinparam defaultFontName monospaced")
    lines.append("skinparam defaultFontColor white")
    lines.append("skinparam defaultFontSize 11")
    lines.append("skinparam state {")
    lines.append("  BackgroundColor #1e293b")
    lines.append("  BorderColor #3b82f6")
    lines.append("  FontColor white")
    lines.append("  ArrowColor #94a3b8")
    lines.append("}")
    lines.append("")
    lines.append("title Comprehensive Facility Condition Markov Model")
    lines.append("note top")
    lines.append("  **FCI (Facility Condition Index)**: 0-100 scale measuring system condition")
    lines.append("  • Good (67-100%): Fully functional, minimal issues, meets all requirements")
    lines.append("  • Fair (34-66%): Functional but degraded, needs attention, some risk")
    lines.append("  • Poor (0-33%): Significant problems, operational risk, failure likely")
    lines.append("  **System failures cascade**: Structural failures affect all systems above")
    lines.append("  **Operational impact**: Poor condition = reduced capacity, safety risks, higher costs")
    lines.append("end note")
    lines.append("")

    lines.append("' === SYSTEM STATES WITH MARKOV TRANSITIONS ===")
    for system, life in systems.items():
        sid = slug(system)
        matrix = markov_matrices.get(system)
        if not matrix:
            continue
        
        deg_rate = matrix.get("degradation_rate_per_year", 0)
        g_stay = round(matrix["matrix"][0][0] * 100, 1)
        g_to_f = round(matrix["matrix"][0][1] * 100, 1)
        f_to_g = round(matrix["matrix"][1][0] * 100, 1)
        f_stay = round(matrix["matrix"][1][1] * 100, 1)
        f_to_p = round(matrix["matrix"][1][2] * 100, 1)
        p_to_f = round(matrix["matrix"][2][1] * 100, 1)
        p_stay = round(matrix["matrix"][2][2] * 100, 1)
        
        desc = system_descriptions.get(system) if system_descriptions else get_system_description_legacy(system)
        deps = get_system_dependencies(system, list(systems.keys()))
        
        lines.append(f"")
        lines.append(f"state \"{escape_label(system)}\" as {sid} {{")
        lines.append(f"  {sid} : **Life Expectancy** = {life} years")
        lines.append(f"  {sid} : **Degradation Rate** = {deg_rate}% per year")
        lines.append(f"  ")
        lines.append(f"  note right of {sid}")
        if desc and isinstance(desc, dict):
            lines.append(f"    Good (67-100% FCI):")
            lines.append(f"    {escape_label(desc.get('good', 'Fully functional'))}")
            lines.append(f"    ")
            lines.append(f"    Fair (34-66% FCI):")
            lines.append(f"    {escape_label(desc.get('fair', 'Functional but degraded'))}")
            lines.append(f"    ")
            lines.append(f"    Poor (0-33% FCI):")
            lines.append(f"    {escape_label(desc.get('poor', 'Significant problems'))}")
            lines.append(f"    ")
            lines.append(f"    Impact: {escape_label(desc.get('impact', 'Affects facility operations'))}")
            if desc.get('importance'):
                lines.append(f"    Importance: {desc.get('importance', 0):.0f}%")
            if desc.get('failure_year'):
                lines.append(f"    Predicted Failure: Year {desc.get('failure_year')}")
            elif desc.get('remaining_years'):
                lines.append(f"    Remaining Life: {desc.get('remaining_years', 0):.1f} years")
        else:
            lines.append(f"    System: {escape_label(system)}")
            lines.append(f"    Life Expectancy: {life} years")
        if deps:
            deps_str = ', '.join([escape_label(d) for d in deps[:3]])
            if len(deps) > 3:
                deps_str += f" +{len(deps)-3} more"
            lines.append(f"    Affects Systems: {deps_str}")
        lines.append(f"  end note")
        lines.append(f"  ")
        failure_info = ""
        if failure_predictions and system in failure_predictions:
            pred = failure_predictions[system]
            if pred.get("failure_year"):
                failure_info = f" | Predicted Failure: {pred['failure_year']}"
            elif pred.get("remaining_years"):
                failure_info = f" | Remaining: {pred['remaining_years']:.1f}yr"
        
        lines.append(f"  state \"GOOD [67-100% FCI]{failure_info}\" as {sid}_G #90EE90")
        lines.append(f"  state \"FAIR [34-66% FCI]{failure_info}\" as {sid}_F #FFD700")
        lines.append(f"  state \"POOR [0-33% FCI]{failure_info}\" as {sid}_P #FF6347")
        lines.append(f"  state \"FAILED\" as {sid}_X #8B0000")
        lines.append(f"  ")
        lines.append(f"  {sid}_G --> {sid}_G : Stay {g_stay}%")
        lines.append(f"  {sid}_G --> {sid}_F : Degrade {g_to_f}%")
        lines.append(f"  {sid}_F --> {sid}_G : Improve {f_to_g}%")
        lines.append(f"  {sid}_F --> {sid}_F : Stay {f_stay}%")
        lines.append(f"  {sid}_F --> {sid}_P : Degrade {f_to_p}%")
        lines.append(f"  {sid}_P --> {sid}_F : Repair {p_to_f}%")
        lines.append(f"  {sid}_P --> {sid}_P : Stay {p_stay}%")
        lines.append(f"  {sid}_P --> {sid}_X : Fail")
        lines.append(f"}}")
        
        if deps:
            for dep_sys in deps[:5]:
                dep_sid = slug(dep_sys)
                lines.append(f"{sid}_P -[#FF6347,dashed]-> {dep_sid}_P : Failure Risk")

    lines.append("")
    lines.append("' === FACILITIES AND SITES ===")
    facilities_map: Dict[str, List] = {}
    
    for year, plist in sorted(projects_by_year.items()):
        for idx, proj in enumerate(plist):
            scope = proj.get("scope", "") or ""
            title = proj.get("title", "") or f"Project_{year}_{idx}"
            cost_text = proj.get("cost_text", "") or "$0"
            
            site_match = re.search(r"(Site [A-J]|Building \d+)", scope, re.IGNORECASE)
            site_name = site_match.group(1) if site_match else f"Site_Unknown"
            
            if site_name not in facilities_map:
                facilities_map[site_name] = []
            
            target_systems = identify_systems_from_scope(scope, list(systems.keys()))
            if not target_systems:
                target_systems = ["All_Systems"]
            
            facilities_map[site_name].append({
                "year": year,
                "idx": idx,
                "title": title,
                "cost": cost_text,
                "systems": target_systems,
                "scope_preview": scope[:80].replace('\n', ' ')
            })

    for site_name, projects in facilities_map.items():
        site_id = slug(site_name)
        lines.append(f"")
        lines.append(f"state \"{escape_label(site_name)}\" as {site_id}_FAC #4A5568 {{")
        lines.append(f"  {site_id}_FAC : {len(projects)} project(s)")
        
        projects_by_yr = {}
        for p in projects:
            yr = p["year"]
            if yr not in projects_by_yr:
                projects_by_yr[yr] = []
            projects_by_yr[yr].append(p)
        
        for yr in sorted(projects_by_yr.keys()):
            yr_projects = projects_by_yr[yr]
            lines.append(f"  ")
            lines.append(f"  state \"FY{yr}\" as {site_id}_Y{yr} {{")
            
            for p in yr_projects:
                proj_id = f"{site_id}_P{yr}_{p['idx']}"
                title_clean = escape_label(p['title'])
                cost_clean = escape_label(p['cost'])
                systems_str = ', '.join([s[:15] for s in p['systems'][:3]])
                if len(p['systems']) > 3:
                    systems_str += f" +{len(p['systems'])-3} more"
                
                lines.append(f"    state \"{title_clean}\" as {proj_id} #3B82F6")
                lines.append(f"    {proj_id} : Cost: {cost_clean}")
                lines.append(f"    {proj_id} : Systems: {escape_label(systems_str)}")
                lines.append(f"    {proj_id} : Year: {yr}")
            
            lines.append(f"  }}")
        
        lines.append(f"}}")

    lines.append("")
    lines.append("' === PROJECT TO SYSTEM CONNECTIONS ===")
    for site_name, projects in facilities_map.items():
        site_id = slug(site_name)
        for p in projects:
            proj_id = f"{site_id}_P{p['year']}_{p['idx']}"
            systems_affected = p['systems']
            
            if "All_Systems" in systems_affected:
                systems_affected = list(systems.keys())
            
            for sys in systems_affected:
                sid = slug(sys)
                lines.append(f"{proj_id} -[#00FF00]-> {sid}_G : Restore")

    lines.append("")
    years_list = sorted(projects_by_year.keys())
    if years_list:
        lines.append(f"note bottom")
        lines.append(f"  Timeline: FY{years_list[0]} to FY{years_list[-1]}")
        lines.append(f"  Total Projects: {sum(len(plist) for plist in projects_by_year.values())}")
        lines.append(f"  Total Systems: {len(systems)}")
        lines.append(f"end note")

    lines.append("")
    lines.append("@enduml")
    return "\n".join(lines)


def run_simulation(
    systems: Dict[str, int],
    projects_by_year: Dict[int, List[Dict]],
    years: int | None = None,
    stop_at_fci: float = 0.0,
    max_years: int = 120,
) -> Dict:
    if projects_by_year:
        current_year = min(projects_by_year.keys())
        max_project_year = max(projects_by_year.keys())
        if years is not None:
            num_years = min(years, max_years)
        else:
            num_years = min(max_project_year - current_year + 10, max_years)
    else:
        current_year = 2024
        num_years = min(years if years is not None else 50, max_years)
    
    system_names = list(systems.keys())
    num_systems = len(system_names)
    
    if num_systems == 0:
        return {
            "unfunded": [],
            "funded": [],
            "system_predictions_unfunded": {},
            "system_predictions_funded": {},
            "stop_reason": "no_systems",
            "years_simulated": 0,
            "start_year": current_year,
        }
    
    life_expectancies = np.array([systems[name] for name in system_names], dtype=np.float64)
    degradation_rates = np.where(life_expectancies > 0, 100.0 / life_expectancies, 0.0)
    
    initial_condition = np.full(num_systems, 100.0, dtype=np.float64)
    
    years_array = np.arange(current_year, current_year + num_years, dtype=np.int32)
    
    conditions_unfunded = np.zeros((num_systems, num_years), dtype=np.float64)
    conditions_unfunded[:, 0] = initial_condition
    
    conditions_funded = np.zeros((num_systems, num_years), dtype=np.float64)
    conditions_funded[:, 0] = initial_condition
    
    project_mask = np.zeros((num_systems, num_years), dtype=bool)
    
    for year_idx, year in enumerate(years_array):
        if year in projects_by_year:
            for project in projects_by_year[year]:
                scope = project.get("scope", "")
                identified_systems = identify_systems_from_scope(scope, system_names)
                
                if identified_systems:
                    system_indices = [system_names.index(s) for s in identified_systems if s in system_names]
                    if system_indices:
                        project_mask[system_indices, year_idx] = True
                else:
                    project_mask[:, year_idx] = True
    
    year_indices = np.arange(num_years, dtype=np.float64)
    
    cumulative_degradation = np.outer(degradation_rates, year_indices)
    
    degraded_conditions = initial_condition[:, np.newaxis] - cumulative_degradation
    
    degraded_conditions = np.clip(degraded_conditions, 0.0, 100.0)
    
    conditions_unfunded = degraded_conditions.copy()
    
    year_indices_2d = np.tile(np.arange(num_years, dtype=np.int32), (num_systems, 1))
    restore_year_matrix = np.where(project_mask, year_indices_2d, -1)
    last_restore_year_matrix = np.maximum.accumulate(restore_year_matrix, axis=1)
    
    years_since_restore = year_indices_2d - np.maximum(last_restore_year_matrix, 0)
    years_since_restore = np.maximum(years_since_restore, 0)
    
    degradation_since_restore = np.outer(degradation_rates, np.ones(num_years)) * years_since_restore
    
    was_restored = last_restore_year_matrix >= 0
    base_condition = np.where(was_restored, 100.0 - degradation_since_restore, degraded_conditions)
    
    conditions_funded = np.clip(base_condition, 0.0, 100.0)
    conditions_funded[project_mask] = 100.0
    
    avg_fci_unfunded = np.mean(conditions_unfunded, axis=0)
    avg_fci_funded = np.mean(conditions_funded, axis=0)
    
    stop_condition = (avg_fci_unfunded <= stop_at_fci) & (avg_fci_funded <= stop_at_fci)
    stop_idx = np.argmax(stop_condition) if np.any(stop_condition) else num_years
    stop_idx = min(stop_idx + 1, num_years) if stop_idx > 0 else num_years
    
    conditions_unfunded = conditions_unfunded[:, :stop_idx]
    conditions_funded = conditions_funded[:, :stop_idx]
    years_array = years_array[:stop_idx]
    avg_fci_unfunded = avg_fci_unfunded[:stop_idx]
    avg_fci_funded = avg_fci_funded[:stop_idx]
    
    scenario_a_results = [
        {"year": int(year), "fci": round(float(fci), 2)}
        for year, fci in zip(years_array, avg_fci_unfunded)
    ]
    scenario_b_results = [
        {"year": int(year), "fci": round(float(fci), 2)}
        for year, fci in zip(years_array, avg_fci_funded)
    ]
    
    system_histories_unfunded = {}
    system_histories_funded = {}
    
    for sys_idx, system_name in enumerate(system_names):
        system_histories_unfunded[system_name] = [
            {
                "year": int(year),
                "fci": round(float(fci), 2),
                "state": get_condition_state(float(fci))
            }
            for year, fci in zip(years_array, conditions_unfunded[sys_idx, :])
        ]
        system_histories_funded[system_name] = [
            {
                "year": int(year),
                "fci": round(float(fci), 2),
                "state": get_condition_state(float(fci))
            }
            for year, fci in zip(years_array, conditions_funded[sys_idx, :])
        ]
    
    stop_reason = "max_years" if stop_idx >= max_years else "fci_threshold"
    
    return {
        "unfunded": scenario_a_results,
        "funded": scenario_b_results,
        "system_predictions_unfunded": system_histories_unfunded,
        "system_predictions_funded": system_histories_funded,
        "stop_reason": stop_reason,
        "years_simulated": len(scenario_a_results),
        "start_year": int(current_year),
    }


def get_condition_state(fci: float) -> str:
    if fci >= 67:
        return "Good"
    elif fci >= 34:
        return "Fair"
    else:
        return "Poor"


@app.post("/predict")
async def predict(
    projects_file: UploadFile = File(...),
    inventory_file: UploadFile = File(...),
):
    try:
        projects_df = parse_file(projects_file)
        inventory_df = parse_file(inventory_file)

        systems = extract_systems_and_life_expectancy(inventory_df)
        if not systems:
            raise HTTPException(
                status_code=400,
                detail="Could not extract systems from inventory file. Please check the file format.",
            )

        facilities = extract_facilities_and_systems(inventory_df)
        
        projects_by_year = extract_projects(projects_df)
        
        results = run_simulation(systems, projects_by_year, years=None, stop_at_fci=0.0, max_years=120)
        
        degradation_rates = {
            system: calculate_degradation_rate(life_exp)
            for system, life_exp in systems.items()
        }
        
        start_year = results.get("start_year", 2024)
        current_fci = {}
        if results.get("system_predictions_unfunded"):
            for system in systems.keys():
                system_history = results["system_predictions_unfunded"].get(system, [])
                if system_history:
                    current_fci[system] = system_history[0].get("fci", 100.0)
                else:
                    current_fci[system] = 100.0
        
        system_importance = calculate_system_importance(systems, facilities, projects_by_year)
        system_correlations = calculate_system_correlations(systems, facilities, projects_by_year)
        failure_predictions = predict_failure_dates(systems, current_fci, degradation_rates, start_year)
        
        facilities_per_system = {}
        for system in systems.keys():
            count = 0
            for facility_data in facilities.values():
                facility_systems = facility_data.get("systems", [])
                if "all" in [s.lower() for s in facility_systems]:
                    count += 1
                elif any(system.lower() in str(s).lower() for s in facility_systems):
                    count += 1
            facilities_per_system[system] = count
        
        system_descriptions = {}
        for system in systems.keys():
            importance = system_importance.get(system, 0.0)
            correlations = system_correlations.get(system, [])
            failure_pred = failure_predictions.get(system, {})
            facility_count = facilities_per_system.get(system, 0)
            
            system_descriptions[system] = generate_system_description_from_data(
                system, importance, correlations, failure_pred, facility_count
            )

        markov_matrices = {
            system: calculate_markov_transition_matrix(life_exp)
            for system, life_exp in systems.items()
        }

        plantuml_diagram = build_markov_plantuml(
            systems, 
            markov_matrices, 
            projects_by_year, 
            facilities,
            system_descriptions,
            failure_predictions
        )
        plantuml_svg = render_plantuml_svg(plantuml_diagram)

        stats = {
            "projects_rows": int(len(projects_df)),
            "inventory_rows": int(len(inventory_df)),
            "project_years": int(len(projects_by_year)),
            "systems_count": int(len(systems)),
            "facilities_count": int(len(facilities)),
            "avg_life_expectancy": round(sum(systems.values()) / max(len(systems), 1), 1),
        }

        return {
            "success": True,
            "systems": systems,
            "facilities": {name: data for name, data in facilities.items()},
            "projects_by_year": {str(k): v for k, v in projects_by_year.items()},
            "results": results,
            "markov_matrices": markov_matrices,
            "system_importance": system_importance,
            "system_correlations": {k: [{"system": s[0], "score": s[1]} for s in v] for k, v in system_correlations.items()},
            "failure_predictions": failure_predictions,
            "system_descriptions": system_descriptions,
            "plantuml_diagram": plantuml_diagram,
            "plantuml_svg": plantuml_svg,
            "stats": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


@app.post("/render-plantuml")
async def render_plantuml(diagram: dict):
    try:
        diagram_text = diagram.get("diagram_source", "")
        if not diagram_text:
            raise HTTPException(status_code=400, detail="diagram_source is required")
        
        svg = render_plantuml_svg(diagram_text)
        if svg:
            return {"success": True, "svg": svg}
        else:
            raise HTTPException(status_code=500, detail="Failed to render diagram")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rendering diagram: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Sustainment Prediction Dashboard API", "status": "running"}
