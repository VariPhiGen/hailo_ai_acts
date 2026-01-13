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

def fill_form(template_filename, form_values_map, headers=None, timeout=None, post_url=None, max_retries=None):
    """
    Loads the template file, replaces placeholders with form_values_map,
    then makes a POST request to the constant POST_URL with the JSON payload.
    Returns the JSON response.

    Args:
        template_filename (str): Path to JSON template containing placeholders {{key}}.
        form_values_map (dict): Dictionary of replacements for placeholders.
        headers (dict, optional): HTTP headers. Defaults to application/json.
        timeout (int, optional): Request timeout in seconds. Default from API_REQUEST_TIMEOUT env var (30).
        max_retries (int, optional): Maximum number of retry attempts. Default from API_MAX_RETRIES env var (3).
        post_url (str, optional): Custom POST URL. Defaults to DEFAULT_POST_URL.

    Returns:
        dict: JSON response from the POST request.

    Raises:
        Exception on file/read/parse/network errors after all retries exhausted.
    """
    # Set defaults from environment variables if not provided
    if timeout is None:
        timeout = int(os.getenv("API_REQUEST_TIMEOUT", 30))
    if max_retries is None:
        max_retries = int(os.getenv("API_MAX_RETRIES", 3))

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

    import time

    for attempt in range(max_retries):
        try:
            logger.info(f'POST attempt {attempt + 1}/{max_retries} to {target_url}')
            response = requests.post(target_url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            logger.info(f'POST request successful on attempt {attempt + 1}')
            return response.json()

        except requests.exceptions.Timeout as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f'POST request timed out after {max_retries} attempts: {str(e)} (url={target_url})')
                raise Exception(f"POST request timed out after {max_retries} attempts: {str(e)} (url={target_url})")

            # Exponential backoff: wait 2^attempt seconds (1, 2, 4 seconds)
            wait_time = 2 ** attempt
            logger.warning(f'POST timeout on attempt {attempt + 1}, retrying in {wait_time}s: {str(e)}')
            time.sleep(wait_time)

        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f'POST connection failed after {max_retries} attempts: {str(e)} (url={target_url})')
                raise Exception(f"POST connection failed after {max_retries} attempts: {str(e)} (url={target_url})")

            # Exponential backoff for connection issues
            wait_time = 2 ** attempt
            logger.warning(f'POST connection error on attempt {attempt + 1}, retrying in {wait_time}s: {str(e)}')
            time.sleep(wait_time)

        except requests.RequestException as e:
            # For other request exceptions, don't retry
            logger.error(f'POST request failed (non-retryable): {str(e)} (url={target_url})')
            raise Exception(f"POST request failed: {str(e)} (url={target_url})")

    # This should never be reached, but just in case
    raise Exception(f"POST request failed after {max_retries} attempts (url={target_url})")
