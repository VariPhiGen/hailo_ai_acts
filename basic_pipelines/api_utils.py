import json
import requests
from typing import Dict, Any


def load_api_template(template_path: str) -> Dict[str, Any]:
    """Load the API template JSON file."""
    with open(template_path, "r") as f:
        return json.load(f)


def get_endpoint_config(api_template: Dict[str, Any], endpoint_name: str) -> Dict[str, Any]:
    """Retrieve configuration for a specific endpoint."""
    endpoints = api_template.get("endpoints", {})
    if endpoint_name not in endpoints:
        raise KeyError(f"Endpoint '{endpoint_name}' not found in API template.")
    return endpoints[endpoint_name]


def extract_required_data(endpoint_config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only required fields for the given endpoint from input data."""
    required_fields = endpoint_config.get("required_fields", {})
    extracted = {}

    for api_field, data_field in required_fields.items():
        if data_field not in data:
            raise KeyError(f"Missing required field '{data_field}' in input data.")
        extracted[api_field] = data[data_field]
    return extracted


def prepare_headers(endpoint_config: Dict[str, Any], api_key: str = None) -> Dict[str, str]:
    """Prepare headers for API request, replacing placeholders if needed."""
    headers = endpoint_config.get("headers", {})
    prepared = {}

    for k, v in headers.items():
        # Replace {{API_KEY}} only if key is provided
        if isinstance(v, str) and "{{API_KEY}}" in v:
            if api_key:
                v = v.replace("{{API_KEY}}", api_key)
            else:
                # If API key not provided, remove the header entirely
                continue
        prepared[k] = v

    return prepared


def call_api(
    api_template: Dict[str, Any],
    endpoint_name: str,
    data: Dict[str, Any],
    api_key: str = None,
) -> Dict[str, Any]:
    """Call the specified API endpoint using template and data."""
    base_url = api_template["base_url"]
    endpoint_config = get_endpoint_config(api_template, endpoint_name)

    extracted_feilds = extract_required_data(endpoint_config, data)
    headers = prepare_headers(endpoint_config, api_key)
    url = base_url + endpoint_config["path"]
    method = endpoint_config.get("method", "POST").upper()

    print(f"➡️ Calling {method} {url}")
    print(f"📦 Payload: {payload}")
    print(f"🧾 Headers: {headers}")

    if method == "POST":
        response = requests.post(url, json=payload, headers=headers)
    elif method == "GET":
        response = requests.get(url, params=payload, headers=headers)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    try:
        return response.json()
    except Exception:
        return {"status_code": response.status_code, "text": response.text}
