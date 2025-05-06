import time
import openai
import requests
import urllib3
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
import json
import re
from functools import lru_cache

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------------
# ⚠️  API KEYS ARE NOW USER‑PROVIDED AT RUN‑TIME VIA STREAMLIT SIDEBAR INPUTS ⚠️
# -----------------------------------------------------------------------------
# Leave the placeholders empty – they will be overwritten after the user enters
# her keys in the sidebar.  All functions reference these globals at call‑time,
# so updating them in `main()` is sufficient.

AMAP_KEY = ""  # will be filled by st.sidebar.text_input
DYNAMIC_ROUTE_URL = "https://restapi.amap.com/v5/direction/driving?parameters"
STATIC_MAP_URL   = "https://restapi.amap.com/v3/staticmap"
GEOCODE_URL      = "https://restapi.amap.com/v3/geocode/geo"

OPENAI_MODEL = "gpt-4o"
client = openai  # keep the alias used throughout the code
# openai.api_key will be set after the user supplies it in the sidebar

# ─────────────────────────────────────────────────────────────────────────────
# Utility functions that depend on the *current* value of AMAP_KEY. Because the
# key is looked up at call‑time, changing the global later is fine.
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def reverse_geocode_amap(lng: float, lat: float) -> str:
    """Return a formatted Chinese place name for one (lng, lat) via AMap."""
    if not AMAP_KEY:
        return ""  # early return if key not yet available

    url = (
        "https://restapi.amap.com/v3/geocode/regeo"
        f"?key={AMAP_KEY}&location={lng:.6f},{lat:.6f}"
        "&radius=1000&extensions=base&output=JSON"
    )
    try:
        data = requests.get(url, timeout=5).json()
        return data.get("regeocode", {}).get("formatted_address", "")
    except Exception:
        return ""


def geocode(address):
    if not AMAP_KEY:
        return None
    params = {"address": address, "key": AMAP_KEY}
    resp = requests.get(GEOCODE_URL, params=params, verify=False)
    data = resp.json()
    if data.get("status") == "1" and data.get("geocodes"):
        location = data["geocodes"][0].get("location")
        if location:
            lng, lat = location.split(",")
            return {"name": address, "coord": (float(lng), float(lat))}
    return None


def get_all_routes(origin, destination):
    if not AMAP_KEY:
        st.error("请先在侧栏输入 AMap API Key！")
        return None

    origin_data      = geocode(origin)
    destination_data = geocode(destination)
    if not origin_data or not destination_data:
        st.error("地址解析失败")
        return None

    params = {
        "origin":       f"{origin_data['coord'][0]},{origin_data['coord'][1]}",
        "destination":  f"{destination_data['coord'][0]},{destination_data['coord'][1]}",
        "strategy":     32,
        "show_fields":  "cost,tmcs,navi,cities,polyline",
        "key":          AMAP_KEY,
    }

    resp = requests.get(DYNAMIC_ROUTE_URL, params=params, verify=False)
    data = resp.json()

    if data.get("status") == "1" and int(data.get("count", 0)) > 0:
        return data["route"]["paths"], origin_data, destination_data
    else:
        st.error("未找到有效路线")
        return None



def _parse_lng_lat(point_str: str) -> tuple[float | None, float | None]:
    """
    Helper: convert 'lng,lat' → (lng, lat) as floats.
    Returns (None, None) on failure.
    """
    try:
        lng_str, lat_str = point_str.split(",")
        return float(lng_str), float(lat_str)
    except Exception:
        return None, None


def _parse_point(point_str: str) -> tuple[float | None, float | None]:
    try:
        lng_str, lat_str = point_str.split(",")
        return float(lng_str), float(lat_str)
    except Exception:
        return None, None


def create_routes_and_segments(
    paths,
    origin_info: dict | None = None,
    dest_info: dict | None = None,
    *,
    include_place_names: bool = False,        # still controls route-level names
):
    """
    Build (routes_df, segments_df) from AMap `paths`.

    • Route-level origin_name / dest_name are added only if
      include_place_names=True.
    • Segment-level start_place / end_place are **always** added.
    """
    # -------------------------------------------------------------------------
    if not paths:
        return pd.DataFrame(), pd.DataFrame()

    first_route = paths[0]
    route_name  = "Route_1"
    steps       = first_route.get("steps", [])

    # ---------- coords for the whole route -----------------------------------
    if steps:
        o_lng, o_lat = _parse_point(steps[0].get("polyline", "").split(";")[0])
        d_lng, d_lat = _parse_point(steps[-1].get("polyline", "").split(";")[-1])
    else:
        o_lng = o_lat = d_lng = d_lat = None

    if origin_info:
        o_lng = origin_info.get("lng", o_lng)
        o_lat = origin_info.get("lat", o_lat)
    if dest_info:
        d_lng = dest_info.get("lng", d_lng)
        d_lat = dest_info.get("lat", d_lat)

    origin_name = dest_name = ""
    if include_place_names and o_lng and o_lat and d_lng and d_lat:
        origin_name = reverse_geocode_amap(o_lng, o_lat)
        dest_name   = reverse_geocode_amap(d_lng, d_lat)

    # ================= ROUTE-LEVEL DF ========================================
    route_row = {
        "route_name":       route_name,
        "distance":         first_route.get("distance", 0),
        "duration":         first_route.get("cost", {}).get("duration"),
        "tolls":            first_route.get("cost", {}).get("tolls", 0),
        "all_instructions": "\n".join(s.get("instruction", "") for s in steps),
        "origin_lng":       o_lng,
        "origin_lat":       o_lat,
        "dest_lng":         d_lng,
        "dest_lat":         d_lat,
    }
    if include_place_names:
        route_row.update({"origin_name": origin_name, "dest_name": dest_name})

    routes_df = pd.DataFrame([route_row])

    # ================= SEGMENT-LEVEL DF =======================================
    seg_rows = []
    for idx, step in enumerate(steps, start=1):
        poly = step.get("polyline", "")
        pts  = poly.split(";")
        s_lng, s_lat = _parse_point(pts[0])  if pts else (None, None)
        e_lng, e_lat = _parse_point(pts[-1]) if pts else (None, None)

        seg_rows.append({
            "route_name":    route_name,
            "segment_name":  f"{route_name}_segment_{idx}",
            "instruction":   step.get("instruction", ""),
            "step_distance": step.get("step_distance", ""),
            "polyline":      poly,
            # route-level meta
            "origin_lng":    o_lng,
            "origin_lat":    o_lat,
            "dest_lng":      d_lng,
            "dest_lat":      d_lat,
            # NEW: segment-level start / end coords & place names
            "start_lng":     s_lng,
            "start_lat":     s_lat,
            "start_place":   reverse_geocode_amap(s_lng, s_lat) if s_lng and s_lat else "",
            "end_lng":       e_lng,
            "end_lat":       e_lat,
            "end_place":     reverse_geocode_amap(e_lng, e_lat) if e_lng and e_lat else "",
        })

    segments_df = pd.DataFrame(seg_rows)
    return routes_df, segments_df

def change_route(user_scenario: str, routes_df: pd.DataFrame, segments_df: pd.DataFrame, max_retries=5):
    """
    Sends a prompt to GPT about a user scenario, passing route/segment data as CSV.
    Returns GPT's JSON response (string) or None if fails.
    """
    csv_text = routes_df.to_csv(index=False)
    segment_text = segments_df.to_csv(index=False)

    prompt = f"""
    Below is the content of one route: {csv_text},
    and segment details: {segment_text}.

    Scenario: "{user_scenario}"

    Task:
    1. Determine if the scenario requested for route change.
    2. If there is route change, find all place names mentioned in scenario, return a JSON object using the structure:
    {{
      "origin": "",
      "pathway": "",
      "destination": ""
    }}
       - Fill each field with the exact place name and its city (if city is not mentioned, use original city).
       - If a particular key cannot be determined, leave it as an empty string.
       - If more than one new place is found, map each to "origin", "pathway", or "destination" accordingly.
    3. If the scenario did not request for route change, return False (without quotes or additional text).
    """.strip()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except client.OpenAIError as e:
            st.warning(f"OpenAI error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    return None

def adversarial_plan(user_scenario: str, routes_df: pd.DataFrame, segments_df: pd.DataFrame, max_retries=5):
    """
    Sends a prompt to GPT about a user scenario, passing route/segment data as CSV.
    Returns GPT's JSON response (string) or None if fails.
    """
    csv_text = routes_df.to_csv(index=False)
    segment_text = segments_df.to_csv(index=False)

    prompt = user_prompt = f"""
Below is the content of one route: {csv_text}),
and segment details: {segment_text}.

Scenario: "{user_scenario}"

Task:
1. Identify relevant routes that reflect changes or attacks.
2. For each potential case, produce JSON with keys:
   - from_route
   - to_route
   - segment
3. If the route doesn't change, 'to_route' = 'from_route'.
4. Output only valid JSON in the format without triple quotes:

{{
  "case1": {{
    "from_route": "...",
    "to_route": "...",
    "segment": "..."
  }},
  "case2": {{
    "from_route": "...",
    "to_route": "...",
    "segment": "..."
  }}
}}

No explanations, no code blocks, no extra text.
""".strip()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except client.OpenAIError as e:
            st.warning(f"OpenAI error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    return None


def new_origin(user_scenario: str, routes_df: pd.DataFrame, segments_df: pd.DataFrame, route, max_retries=5):
    """
    Sends a prompt to GPT about a user scenario, passing route/segment data as CSV.
    Returns GPT's JSON response (string) or None if fails.
    """
    csv_text = routes_df.to_csv(index=False)
    segment_text = segments_df.to_csv(index=False)

    prompt = f"""
    Below is the content of one route: {csv_text},
    and segment details: {segment_text}.

    Scenario: "{user_scenario}"
    New place name list: {route}

    Task:
    1. Identify relevant segment that may contain changes or attacks.
    2. From each potential case, extract the place name from the selected segment and determine the origin, pathway point, and destination. 
    If only destination is changed, make the origin the place name extracted from the segment.
    3. Produce JSON with the keys:
       - segment
       - origin
       - pathway
       - destination
    4. Output only valid JSON in the format without triple quotes, keep "" if no element is found for the factor:
    {{
      "case1": {{
        "segment": "...",
        "origin": "...",
        "pathway": "...",
        "destination": "..."
      }},
      "case2": {{
        "segment": "...",
        "origin": "...",
        "pathway": "...",
        "destination": "..."
      }}
    }}

    No explanations, no code blocks, no extra text.
    """.strip()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except client.OpenAIError as e:
            st.warning(f"OpenAI error on attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)

    return None


# ------------------------
# Parse LLM's JSON output:
# ------------------------

def parse_polyline(polyline_str: str):
    """
    Convert a polyline string "108.94706,34.19905;108.946754,34.20008;..."
    into list of (lng, lat) float pairs.
    """
    results = []
    for seg in polyline_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        lng_str, lat_str = seg.split(",")
        results.append((float(lng_str), float(lat_str)))
    return results


def find_attacked_segments(llm_json_str: str, segments_df: pd.DataFrame):
    """
    1. Parse the LLM JSON specifying "segment" under each case.
    2. Retrieve the polyline from segments_df for each segment.
    3. Return a dict with case -> { "segment_name": ..., "points": [(lng, lat), ...] }
    """
    attacked_segments = {}
    try:
        cases_dict = json.loads(llm_json_str)  # parse LLM response
    except json.JSONDecodeError:
        st.warning("LLM returned invalid JSON, cannot parse.")
        return {}

    for case_id, case_data in cases_dict.items():
        seg_name = case_data.get("segment")
        if not seg_name:
            attacked_segments[case_id] = {"error": "No segment found", "points": []}
            continue

        row = segments_df.loc[segments_df["segment_name"] == seg_name]
        if row.empty:
            attacked_segments[case_id] = {"error": f"No such segment: {seg_name}", "points": []}
            continue

        polyline_str = row.iloc[0]["polyline"]
        points = parse_polyline(polyline_str)

        attacked_segments[case_id] = {
            "segment_name": seg_name,
            "points": points
        }
        # print(attacked_segments)

    return attacked_segments

def index_to_label(index):
    index += 2
    label = ''
    while True:
        index, remainder = divmod(index, 26)
        label = chr(65 + remainder) + label
        if index == 0:
            break
        index -= 1
    return label

def pick_marker_points(attacked_segments: dict):
    """
    根据攻击段生成地图标记点，支持任意数量case，用字母作为标记
    """
    color_palette = [
        "0xFFFF00"  # 黄色
    ]

    markers = []
    for case_index, (case_id, seg_data) in enumerate(attacked_segments.items()):
        points = seg_data.get("points", [])
        if not points:
            continue

        # 获取中间点坐标
        mid_idx = len(points) // 2
        lng_c, lat_c = points[mid_idx]

        # 自动循环使用颜色
        color = color_palette[case_index % len(color_palette)]
        label = index_to_label(case_index)  # Use letters: A, B, C...

        # 构建标记字符串
        marker_str = f"mid,{color},{label}:{lng_c:.6f},{lat_c:.6f}"
        markers.append(marker_str)

    return markers


# Alternative: label ALL points as C1, C2...
# def pick_all_points_as_c(attacked_segments: dict):
#     markers = []
#     c_index = 1
#     for case_id, seg_data in attacked_segments.items():
#         for (lng, lat) in seg_data.get("points", []):
#             label = f"C{c_index}"
#             marker_str = f"small,0xFF00FF,{label}:{lng:.6f},{lat:.6f}"
#             markers.append(marker_str)
#             c_index += 1
#     return markers


def plot_single_route_on_static_map(path, origin_info, dest_info, attack_markers=None):
    # 坐标收集与清洗
    coordinates = []
    for step in path.get("steps", []):
        if poly := step.get("polyline", ""):
            if not poly:
                continue
            for seg in poly.split(";"):
                seg = seg.strip()
                if seg:
                    lng_str, lat_str = seg.split(",")
                    coordinates.append((float(lng_str), float(lat_str)))

    if len(coordinates) > 100:
        coordinates = coordinates[::5]

    if len(coordinates) < 2:
        st.warning("该路线点数不足，无法绘制")
        return

    color = "0xFF0000"
    path_points = ";".join(f"{lng:.6f},{lat:.6f}" for lng, lat in coordinates)
    path_param = f"5,{color},1,,:{path_points}"

    center_lng, center_lat = origin_info["coord"]
    base_markers = f"large,0x00FF00,A:{center_lng},{center_lat}|large,0x00FF00,B:{dest_info['coord'][0]},{dest_info['coord'][1]}"

    if attack_markers:
        base_markers += '|' + '|'.join(attack_markers)

    static_params = {
        "key": AMAP_KEY,
        "size": "600*400",
        "zoom": "10",
        "scale": "2",
        "center": f"{center_lng},{center_lat}",
        "markers": base_markers,
        "paths": path_param
    }
    st.write(static_params)

    resp = requests.get(STATIC_MAP_URL, params=static_params, verify=False)
    # print(resp)
    if resp.status_code != 200:
        st.error(f"静态地图请求失败，状态码: {resp.status_code}")
        st.write("返回内容：", resp.text)
        return

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        st.error("静态地图接口返回的不是图片，请检查参数或输出。")
        st.write(resp.text)
        return

    try:
        image = Image.open(BytesIO(resp.content))
        st.image(image, caption="路线示意图")
    except Exception as e:
        st.error(f"地图渲染错误: {e}")


def handle_new_places(llm_json: str):
    """
    Given the JSON from new_origin(), parse each 'case' and,
    depending on which fields are provided (origin, pathway, destination),
    fetch the relevant routes.
    """
    if not llm_json or llm_json.strip() == "False":
        st.write("LLM indicates no new place found (False).")
        return

    try:
        cases = json.loads(llm_json)
    except json.JSONDecodeError:
        st.error("Invalid JSON returned by new_origin()")
        return

    # A container to hold the route-fetch results
    # e.g. routes_for_cases = { "case1": [ (pathsA, origin_info, path_info), (pathsB, path_info, dest_info) ], ... }
    routes_for_cases = {}

    for case_id, case_data in cases.items():
        origin = case_data.get("origin", "").strip()
        pathway = case_data.get("pathway", "").strip()
        destination = case_data.get("destination", "").strip()

        # Skip if everything is empty
        if not (origin or pathway or destination):
            st.write(f"{case_id} has no new places provided.")
            continue

        # Decide which routes to fetch based on which fields are non-empty.
        case_routes = []

        # CASE: origin, pathway, destination all exist
        if origin and pathway and destination:
            res_op = get_all_routes(origin, pathway)         # origin -> pathway
            res_pd = get_all_routes(pathway, destination)    # pathway -> destination
            case_routes.append(("OP", origin, pathway, res_op))
            case_routes.append(("PD", pathway, destination, res_pd))

        # CASE: only origin and destination
        elif origin and destination and not pathway:
            res_od = get_all_routes(origin, destination)
            case_routes.append(("OD", origin, destination, res_od))

        # You can add more logic if only one field is provided, etc.

        routes_for_cases[case_id] = case_routes

    return routes_for_cases


def plot_original_and_changed_route_on_static_map(
    original_path,
    changed_path,
    original_origin_info,
    original_dest_info,
    changed_origin_info,
    changed_dest_info,
    attack_markers=None
):


    def get_coordinates_from_path(path):
        # Collect and optionally thin out the route coordinates
        coords = []
        for step in path.get("steps", []):
            poly = step.get("polyline", "")
            if not poly:
                continue
            for seg in poly.split(";"):
                seg = seg.strip()
                if seg:
                    lng_str, lat_str = seg.split(",")
                    coords.append((float(lng_str), float(lat_str)))
        # Thinning out if too many coordinates
        if len(coords) > 100:
            coords = coords[::5]
        return coords

    # 1) Extract coordinate lists
    original_coords = get_coordinates_from_path(original_path)
    changed_coords = get_coordinates_from_path(changed_path)

    # 2) Basic checks
    if len(original_coords) < 2:
        st.warning("Original route has insufficient points to plot.")
        return
    if len(changed_coords) < 2:
        st.warning("Changed route has insufficient points to plot.")
        return

    # 3) Build the 'paths' parameter for both routes
    #    (the AMap Static Map API supports multiple 'paths' separated by "|")
    def build_path_param(coords, color_hex):
        path_points = ";".join(f"{lng:.6f},{lat:.6f}" for lng, lat in coords)
        # Format: (weight),(color),(transparency),(outline),(sequence of coords)
        return f"5,{color_hex},1,,:{path_points}"

    # Original route in red, changed route in blue
    original_path_param = build_path_param(original_coords, "0xFF0000")  # Red
    changed_path_param = build_path_param(changed_coords, "0x0000FF")   # Blue
    combined_paths = f"{original_path_param}|{changed_path_param}"

    # 4) Define markers for origin/destination of both routes
    #    (You can label them however you like. Here we do A/B for original and C/D for changed.)
    base_markers = (
        f"large,0x00FF00,A:{original_origin_info['coord'][0]},{original_origin_info['coord'][1]}"   # Original origin
        f"|large,0x00FF00,B:{original_dest_info['coord'][0]},{original_dest_info['coord'][1]}"      # Original dest
        f"|large,0xFF00FF,C:{changed_origin_info['coord'][0]},{changed_origin_info['coord'][1]}"    # Changed origin
        f"|large,0xFF00FF,D:{changed_dest_info['coord'][0]},{changed_dest_info['coord'][1]}"        # Changed dest
    )

    # If you have optional attack markers, add them to the markers parameter
    if attack_markers:
        base_markers += '|' + '|'.join(attack_markers)

    # 5) Pick a center to display the map.
    #    We'll center on the original origin, but you can choose differently.
    center_lng, center_lat = original_origin_info["coord"]

    # 6) Build the static map request parameters
    static_params = {
        "key": AMAP_KEY,
        "size": "600*400",
        "zoom": "10",
        "scale": "2",
        "center": f"{center_lng},{center_lat}",
        "markers": base_markers,
        "paths": combined_paths
    }

    # Display these for debugging
    st.write("Static Map Parameters:", static_params)

    # 7) Send request to AMap Static Map
    resp = requests.get(STATIC_MAP_URL, params=static_params, verify=False)
    if resp.status_code != 200:
        st.error(f"静态地图请求失败，状态码: {resp.status_code}")
        st.write("返回内容：", resp.text)
        return

    # 8) Validate content type and display
    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        st.error("静态地图接口返回的不是图片，请检查参数或输出。")
        st.write(resp.text)
        return

    try:
        image = Image.open(BytesIO(resp.content))
        st.image(image, caption="Original & Changed Route Overlay")
    except Exception as e:
        st.error(f"地图渲染错误: {e}")


def predict_attack_for_one_segment(row, scenario, max_retries=5):
    """
    Given a single row (which contains at least 'segment_name' and 'instruction'),
    ask GPT how an adversarial attack might make the vehicle follow this instruction
    for the given scenario.

    Returns a string representing the predicted 'attack' for that segment.
    """

    segment_name = row.get("segment_name", "")
    instruction = row.get("instruction", "")
    route_name = row.get("route_name", "")
    step_distance = row.get("step_distance", "")
    polyline = row.get("polyline", "")

    # 1) Build a per-row prompt
    prompt = f"""
Below is a single changed route segment:

- scenario: {scenario}
- route_name: {route_name}
- segment_name: {segment_name}
- instruction: {instruction}
- step_distance: {step_distance}
- polyline: {polyline}

Task:
Explain how an adversarial attack would manipulate sensors or environment
such that the vehicle follows this instruction. Return only valid JSON with
the structure:

{{
  "attack": "Your text here"
}}

No extra commentary or Markdown.
""".strip()

    # 2) Send the prompt to GPT, retrying on transient errors
    gpt_response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            gpt_response = response.choices[0].message.content.strip()
            break
        except openai.OpenAIError as e:
            time.sleep(2 ** attempt)

    if not gpt_response:
        # If the model doesn't respond, return an empty string
        return ""

    # 3) Parse the returned JSON
    try:
        attack_data = json.loads(gpt_response)
        # Expecting something like {"attack": "..."}
        return attack_data.get("attack", "")
    except json.JSONDecodeError:
        # If GPT didn't return valid JSON, just return raw text or empty
        return ""

def predict_adversarial_attack_per_row(
    scenario: str, changed_segments_df: pd.DataFrame
) -> pd.DataFrame:

    # Ensure we have an 'attack' column
    if "attack" not in changed_segments_df.columns:
        changed_segments_df["attack"] = ""

    # 1) Iterate through each row
    for idx, row in changed_segments_df.iterrows():
        attack_text = predict_attack_for_one_segment(row, scenario)
        # 2) Update the 'attack' column
        changed_segments_df.at[idx, "attack"] = attack_text
        print(attack_text)

    return changed_segments_df


def predict_attack_for_segment(segment_row, scenario, max_retries=5):
    segment_name = segment_row.get("segment_name", "")
    instruction = segment_row.get("instruction", "")
    step_distance = segment_row.get("step_distance", "")
    route_name = segment_row.get("route_name", "")

    # Build the prompt for GPT
    prompt = f"""
Scenario: "{scenario}"

Segment Info:
- route_name: {route_name}
- segment_name: {segment_name}
- step_distance: {step_distance}

Task:
Explain how an adversarial attack on this segment's environment or sensors
would cause or relate to the scenario. Return ONLY attack description.
""".strip()

    gpt_response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            gpt_response = response.choices[0].message.content.strip()
            break
        except openai.OpenAIError as e:
            time.sleep(2 ** attempt)

    if not gpt_response:
        return ""
    print(gpt_response)
    return gpt_response


def predict_attack_for_attacked_segments(scenario, attacked_data, segments_df):
    attack_predictions = {}

    for case_id, info in attacked_data.items():
        seg_name = info.get("segment_name", "")
        # Attempt to locate that segment in segments_df
        row_match = segments_df.loc[segments_df["segment_name"] == seg_name]

        if row_match.empty:
            # Can't find the segment row in the DF, skip or store an error
            attack_predictions[case_id] = {
                "segment_name": seg_name,
                "attack": "Segment not found in segments_df"
            }
            continue

        # If multiple rows match, just take the first
        segment_row = row_match.iloc[0]

        # Now call the GPT function for that segment
        attack_desc = predict_attack_for_segment(segment_row, scenario)
        print(attack_desc)
        attack_predictions[case_id] = {
            "segment_name": seg_name,
            "attack": attack_desc
        }

    return attack_predictions

def evaluation(segments_df, updated_segments_df, cus_scenario, max_retries=5):
    prompt = f"""
You are an expert route-planning auditor.  
Your job is to decide whether a *changed* plan now satisfies the customer’s scenario.

────────────────────────────────────────
CUSTOMER SCENARIO
{cus_scenario}
────────────────────────────────────────
ORIGINAL PLAN  (each row is a segment)
{segments_df.to_json(orient="records", indent=2)}
────────────────────────────────────────
UPDATED PLAN  (each row is a segment), it will starts from the segment that needs update
{updated_segments_df.to_json(orient="records", indent=2)}
────────────────────────────────────────

**Instructions**

1. Evaluate the UPDATED plan against every explicit and implicit requirement in the CUSTOMER SCENARIO. 
1.1. For location-related requirements, you should compare place against places from CUSTOMER SCENARIO not original plan. a close enough (rough) match counts as compliant.
2. If all requirements are satisfied, output "satisfied": "yes"; otherwise output "satisfied": "no".
3. In one brief sentence, state the single most important reason for your decision.

**Output**

Return **ONLY** valid JSON exactly like:

{{
  "satisfied": "yes" | "no",
  "reason": "<one concise sentence>"
}}
""".strip()

    # 2) Send the prompt to GPT, retrying on transient errors
    gpt_response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            gpt_response = response.choices[0].message.content.strip()
            break
        except openai.OpenAIError:
            time.sleep(2 ** attempt)

    if not gpt_response:
        return {"satisfied": "error", "reason": "No response from model"}

    return gpt_response
    # except json.JSONDecodeError:
    #     # Return raw text so you can inspect what went wrong
    #     return {
    #         "satisfied": "error",
    #         "reason": f"Could not parse JSON: {gpt_response}"
    #     }

def main():
    st.set_page_config(page_title="Adversarial Single‑Route Demo", layout="wide")
    st.title("Adversarial Plan (Only First Route) with Attack Points")

    # ── 1. Collect API keys from the user ────────────────────────────────────
    st.sidebar.header("🔑 API Keys")
    amap_key_input = st.sidebar.text_input("AMap API Key", type="password")
    openai_key_input = st.sidebar.text_input("OpenAI API Key", type="password")

    # Update globals so that helper functions pick up the new keys
    if amap_key_input:
        globals()["AMAP_KEY"] = amap_key_input.strip()
    if openai_key_input:
        openai.api_key = openai_key_input.strip()

    if not AMAP_KEY:
        st.sidebar.info("请输入 AMap API Key 才能调用高德接口。")
    if not openai.api_key:
        st.sidebar.info("请输入 OpenAI API Key 才能调用 GPT‑4o。")

    # Initialize session placeholders
    if "routes_df" not in st.session_state:
        st.session_state.routes_df = None
    if "segments_df" not in st.session_state:
        st.session_state.segments_df = None
    if "amap_path" not in st.session_state:
        st.session_state.amap_path = None
    if "origin_info" not in st.session_state:
        st.session_state.origin_info = None
    if "dest_info" not in st.session_state:
        st.session_state.dest_info = None

    # 1) Form to fetch route
    with st.form("route_form"):
        origin = st.text_input("Origin", "西安市长安大学渭水校区")
        destination = st.text_input("Destination", "西安钟楼")
        submitted = st.form_submit_button("Fetch Routes")

    if submitted:
        with st.spinner("Querying AMap for routes..."):
            result = get_all_routes(origin, destination)
            if result:
                all_paths, origin_data, dest_data = result
                if all_paths:
                    # store the entire route in session for plotting
                    st.session_state.amap_path = all_paths[0]
                    st.session_state.origin_info = origin_data
                    st.session_state.dest_info = dest_data

                    # Create the DataFrames for that route
                    routes_df, segments_df = create_routes_and_segments(all_paths)
                    st.session_state.routes_df = routes_df
                    st.session_state.segments_df = segments_df

                    st.success("Route data fetched and stored in session state!")
                else:
                    st.error("No routes returned from AMap.")
            else:
                st.error("Error retrieving routes from AMap.")

    # Display route & segment info
    if st.session_state.routes_df is not None and not st.session_state.routes_df.empty:
        st.subheader("Single Route Overview")
        st.dataframe(st.session_state.routes_df)

    if st.session_state.segments_df is not None and not st.session_state.segments_df.empty:
        st.subheader("Segment Info")
        st.dataframe(st.session_state.segments_df)

    # 2) Analyze scenariost
    st.subheader("Adversarial Scenario")
    user_scenario = st.text_input("Enter your scenario (e.g. sudden route change)", '车并未按照原定计划的路线行驶，而是从西安钟楼开到了西安大慈恩寺')
    if st.button("Analyze Scenario"):
        routes_df = st.session_state.routes_df
        segments_df = st.session_state.segments_df
        path = st.session_state.amap_path
        origin_info = st.session_state.origin_info
        dest_info = st.session_state.dest_info

        if routes_df is None or segments_df is None or routes_df.empty or segments_df.empty:
            st.warning("No route data available. Fetch routes first.")
        else:
            with st.spinner("Calling GPT for adversarial plan..."):
                route = change_route(user_scenario, routes_df, segments_df)
                print(route)
                if route == "False":
                    llm_json = adversarial_plan(user_scenario, routes_df, segments_df)
                    if llm_json:
                        st.subheader("GPT Output (JSON)")
                        st.write(llm_json)

                        attacked_data = find_attacked_segments(llm_json, segments_df)
                        if attacked_data:
                            c_markers = pick_marker_points(attacked_data)
                            plot_single_route_on_static_map(
                                path,
                                origin_info,
                                dest_info,
                                attack_markers=c_markers
                            )

                            # Now PREDICT the attack for each segment
                            attack_dict = predict_attack_for_attacked_segments(
                                scenario=user_scenario,
                                attacked_data=attacked_data,
                                segments_df=segments_df
                            )
                            st.subheader("Predicted Attacks for Each Case")
                            st.write(attack_dict)

                        else:
                            st.warning("No attacked segment found or JSON invalid.")
                    else:
                        st.error("No response or error from GPT.")
                else:
                    llm_json = new_origin(user_scenario, routes_df, segments_df, route)
                    # print(llm_json)
                    routes_for_cases = handle_new_places(llm_json)
                    # print(routes_for_cases)
                    if routes_for_cases:
                        try:
                            llm_cases_data = json.loads(llm_json)
                        except json.JSONDecodeError:
                            st.warning("Invalid JSON from LLM; cannot reorder changed segment.")
                            llm_cases_data = {}

                        for case_id, route_list in routes_for_cases.items():
                            st.write(f"**{case_id}**:")

                            changed_segment_name = ""
                            case_data = llm_cases_data.get(case_id, {})
                            if case_data:
                                changed_segment_name = case_data.get("segment", "").strip()

                            for (route_type, from_place, to_place, res) in route_list:
                                st.write(f"- {route_type} → from {from_place} to {to_place}")

                                if res is not None:
                                    # 'res' is (paths, from_info, to_info)
                                    paths, from_info, to_info = res
                                    st.write(f"Found {len(paths)} path(s).")

                                    # Build DataFrames for the changed route
                                    routes_df2, segments_df2 = create_routes_and_segments(paths)

                                    # (Optional) Reorder 'segments_df2' so changed_segment is first
                                    if changed_segment_name:
                                        idx_list = segments_df2.index[
                                            segments_df2["segment_name"] == changed_segment_name
                                            ].tolist()
                                        if idx_list:
                                            first_idx = idx_list[0]
                                            new_order = [first_idx] + [
                                                i for i in segments_df2.index if i != first_idx
                                            ]
                                            segments_df2 = segments_df2.loc[new_order].reset_index(drop=True)

                                    # Plot the changed route
                                    if st.session_state.amap_path and st.session_state.origin_info and st.session_state.dest_info:
                                        plot_original_and_changed_route_on_static_map(
                                            original_path=st.session_state.amap_path,
                                            changed_path=paths[0],
                                            original_origin_info=st.session_state.origin_info,
                                            original_dest_info=st.session_state.dest_info,
                                            changed_origin_info=from_info,
                                            changed_dest_info=to_info,
                                            attack_markers=None
                                        )

                                    updated_segments_df = predict_adversarial_attack_per_row(user_scenario,
                                                                                             segments_df2)

                                    # Now 'updated_segments_df' contains an 'attack' column with GPT results
                                    st.subheader("Updated Changed Route Segments with Per-Row Attack Predictions")
                                    st.dataframe(updated_segments_df)
                                    print(segments_df, updated_segments_df)
                                    evaluate=evaluation(segments_df, updated_segments_df, cus_scenario=user_scenario)
                                    print(evaluate)
                                    st.subheader("Evaluation")
                                    st.write(evaluate)

                                else:
                                    st.write("No routes found or error from AMap.")


if __name__ == "__main__":
    main()