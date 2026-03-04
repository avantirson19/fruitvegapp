import streamlit as st
from PIL import Image
from predict import predict_image

st.set_page_config(page_title="Smart Fruit Quality System", layout="centered")


def split_image_into_regions(image: Image.Image, count: int = 3):
    image = image.convert("RGB")
    width, height = image.size
    regions = []
    step = width // count
    for idx in range(count):
        left = idx * step
        right = width if idx == count - 1 else (idx + 1) * step
        regions.append((idx + 1, image.crop((left, 0, right, height))))
    return regions

# -----------------------------
# Premium Styling
# -----------------------------

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef2f3, #e0f7fa);
}
.card {
    background: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Smart Fruit & Vegetable Quality System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered freshness and classification analysis</div>", unsafe_allow_html=True)

# -----------------------------
# Input Section
# -----------------------------

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Select Image Source</div>", unsafe_allow_html=True)

option = st.radio(
    "Select source",
    ["Upload Image", "Use Camera"],
    horizontal=True,
    label_visibility="collapsed",
)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Use Camera":
    camera_file = st.camera_input("Capture Image")
    if camera_file:
        image = Image.open(camera_file)

st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Tips to increase confidence", expanded=False):
    st.write("- Keep only one fruit/vegetable centered in frame")
    st.write("- Use bright, even lighting (avoid shadows)")
    st.write("- Keep plain background and avoid hands/other objects")
    st.write("- Move camera closer so item fills most of the image")
    st.write("- Avoid blur; hold phone steady before capture")

# -----------------------------
# Prediction Section
# -----------------------------

if image is not None:

    st.image(image, caption="Input Image", width="stretch")

    predicted_class, confidence, is_low_confidence, preprocessing_mode, model_name = predict_image(image)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

    st.markdown(f"### 🥭 Identified as: **{predicted_class.capitalize()}**")
    if is_low_confidence:
        st.warning(
            "Model confidence is low. Try a clearer image with the fruit/vegetable centered "
            "and better lighting."
        )

    # Animated Confidence Bar
    st.markdown("#### Confidence Level")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}% confidence")
    st.caption(f"Preprocessing used: `{preprocessing_mode}`")
    st.caption(f"Model used: `{model_name}`")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # Quality Section
    # -----------------------------

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Quality Assessment</div>", unsafe_allow_html=True)

    if is_low_confidence:
        st.info("ℹ Prediction is uncertain. Please upload another image for a reliable result.")
    elif confidence > 90:
        st.success("✅ High freshness detected. Safe and recommended for consumption.")
    elif 70 < confidence <= 90:
        st.warning("⚠ Moderate confidence. Please visually inspect before consuming.")
    else:
        st.error("❌ Low confidence. Consider rechecking or avoiding consumption.")

    st.markdown("</div>", unsafe_allow_html=True)

    if predicted_class == "apple":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Multi-Apple Edibility (Beta)</div>", unsafe_allow_html=True)
        st.caption("Checks up to 3 left-to-right regions. Use a photo where apples are clearly separated.")

        for idx, region in split_image_into_regions(image, count=3):
            item_class, item_confidence, _, _, _ = predict_image(region)
            if item_class == "apple" and item_confidence >= 60:
                st.success(f"Apple {idx}: Likely edible ({item_confidence:.2f}% confidence)")
            elif item_class == "apple":
                st.warning(f"Apple {idx}: Apple detected but quality uncertain ({item_confidence:.2f}%)")
            else:
                st.error(
                    f"Item {idx}: Not confidently detected as apple "
                    f"(detected {item_class}, {item_confidence:.2f}%)"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("AI Model: Optimized GoogLeNet (InceptionV3) | Built using TensorFlow & Streamlit")
