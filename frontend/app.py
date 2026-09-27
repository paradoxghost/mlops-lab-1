"""Upload an image to the Food-11 inference API from Streamlit."""

import math
import os

import requests
import streamlit as st


INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000").rstrip("/")


def request_prediction(name: str, content: bytes, content_type: str) -> dict:
    response = requests.post(
        f"{INFERENCE_URL}/predict",
        files={"file": (name, content, content_type)},
        timeout=(5, 60),
    )
    response.raise_for_status()
    result = response.json()
    if (not isinstance(result, dict) or not isinstance(result.get("category"), str)
            or not isinstance(result.get("confidence"), (int, float))
            or not math.isfinite(result["confidence"])
            or not 0 <= result["confidence"] <= 1):
        raise ValueError("The inference response did not contain a category and confidence")
    return result


def main() -> None:
    st.title("Food-11 classifier")
    uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return
    content = uploaded.getvalue()
    if len(content) > 10 * 1024 * 1024:
        st.error("Please upload an image smaller than 10 MiB.")
        return
    try:
        st.image(content, width=300)
        with st.spinner("Classifying your image..."):
            result = request_prediction(uploaded.name, content, uploaded.type or "application/octet-stream")
        st.write(f"**Prediction:** {result['category']} ({result['confidence']:.1%})")
    except requests.Timeout:
        st.error("Inference timed out. Please try again shortly.")
    except requests.ConnectionError:
        st.error("The inference service is starting or unavailable. Please try again shortly.")
    except requests.HTTPError as error:
        st.error(f"Inference service returned HTTP {error.response.status_code}. Check that the image is valid.")
    except (requests.RequestException, ValueError, OSError):
        st.error("The image or inference response could not be read. Please try another image.")


if __name__ == "__main__":
    main()
