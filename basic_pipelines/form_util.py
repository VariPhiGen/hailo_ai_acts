import json
import re
import logging
import os
import requests

logger = logging.getLogger(__name__)

# Constant URL declared here (overridable via env API_POST_URL)
DEFAULT_POST_URL = os.getenv("API_POST_URL", "http://localhost:80/api/client/1825/mfb/forms_data")

template_cache = {}

def load_template(template_filename):
    if template_filename not in template_cache:
        try:
            with open(template_filename, 'r', encoding='utf-8') as f:
                template_cache[template_filename] = f.read()
        except FileNotFoundError:
            raise Exception(f"Template file '{template_filename}' not found")
        except Exception as e:
            logger.error('Error loading template:', exc_info=e)
            raise Exception(f"Failed to load template {template_filename}: {str(e)}")
    return template_cache[template_filename]

def fill_form(template_filename, form_values_map, headers=None, timeout=10, post_url=None):
    """
    Loads the template file, replaces placeholders with form_values_map,
    then makes a POST request to the constant POST_URL with the JSON payload.
    Returns the JSON response.

    Args:
        template_filename (str): Path to JSON template containing placeholders {{key}}.
        form_values_map (dict): Dictionary of replacements for placeholders.
        headers (dict, optional): HTTP headers. Defaults to application/json.
        timeout (int, optional): Request timeout in seconds. Default 10.

    Returns:
        dict: JSON response from the POST request.

    Raises:
        Exception on file/read/parse/network errors.
    """
    template_str = load_template(template_filename)
    try:
        processed_template = template_str
        for key, value in form_values_map.items():
            placeholder = r'\{\{' + re.escape(key) + r'\}\}'
            processed_template = re.sub(placeholder, str(value), processed_template)

        payload = json.loads(processed_template)
    except Exception as e:
        logger.error('Error processing template:', exc_info=e)
        raise Exception(f"Failed to process JSON template: {str(e)}")

    if headers is None:
        headers = {'Content-Type': 'application/json'}

    target_url = post_url or DEFAULT_POST_URL

    try:
        response = requests.post(target_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error('POST request failed:', exc_info=e)
        raise Exception(f"POST request failed: {str(e)} (url={target_url})")
