from api_utils import load_api_template, call_api

if __name__ == "__main__":
    # Load template
    api_template = load_api_template("api_template.json")

    # Example data
    data = {
        "timestamp_data": "2025-10-21T18:45:00Z",
        "camera_id": "CAM_001",
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    }

    # Option 1: With API key
    response = call_api(api_template, "upload_data", data, api_key="my_secret_token")

    # Option 2: Without API key
    # response = call_api(api_template, "upload_data", data)

    print("✅ API Response:", response)
