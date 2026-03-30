import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Smart Fruit Quality System", layout="wide")

from predict import get_model_status, predict_image


def _sanitize_boxes(boxes, width, height, min_area_ratio=0.008):
    min_area = max(int(width * height * min_area_ratio), 120)
    cleaned = []
    for left, top, right, bottom, area in boxes:
        if area >= min_area and right - left >= 18 and bottom - top >= 18:
            cleaned.append((left, top, right, bottom, area))
    return cleaned


def _dilate_mask(mask, radius=1):
    dilated = mask.copy()
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            if shift_x == 0 and shift_y == 0:
                continue
            source = np.zeros_like(mask)
            y_src_start = max(0, -shift_y)
            y_src_end = mask.shape[0] - max(0, shift_y)
            x_src_start = max(0, -shift_x)
            x_src_end = mask.shape[1] - max(0, shift_x)
            y_dst_start = max(0, shift_y)
            y_dst_end = y_dst_start + (y_src_end - y_src_start)
            x_dst_start = max(0, shift_x)
            x_dst_end = x_dst_start + (x_src_end - x_src_start)
            source[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = mask[y_src_start:y_src_end, x_src_start:x_src_end]
            dilated |= source
    return dilated


def _connected_components(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0

            while stack:
                cy, cx = stack.pop()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)

                for ny, nx in (
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                    (cy - 1, cx - 1),
                    (cy - 1, cx + 1),
                    (cy + 1, cx - 1),
                    (cy + 1, cx + 1),
                ):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            boxes.append((min_x, min_y, max_x + 1, max_y + 1, area))

    return boxes


def _merge_close_boxes(boxes, gap=8):
    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        next_boxes = []
        while merged:
            current = merged.pop(0)
            cl, ct, cr, cb, ca = current
            merged_any = False
            for idx, other in enumerate(merged):
                ol, ot, or_, ob, oa = other
                horizontal_gap = max(0, max(ol - cr, cl - or_))
                vertical_gap = max(0, max(ot - cb, ct - ob))
                if horizontal_gap <= gap and vertical_gap <= gap:
                    current = (min(cl, ol), min(ct, ot), max(cr, or_), max(cb, ob), ca + oa)
                    merged.pop(idx)
                    cl, ct, cr, cb, ca = current
                    merged_any = True
                    changed = True
                    break
            if merged_any:
                merged.insert(0, current)
            else:
                next_boxes.append(current)
        merged = next_boxes
    return merged


def detect_item_regions(image: Image.Image, max_items=6):
    rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    working_width = min(480, original_width)
    working_height = max(1, int(original_height * (working_width / original_width)))
    small = rgb.resize((working_width, working_height))
    arr = np.asarray(small, dtype=np.float32)

    border_pixels = np.concatenate(
        [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]],
        axis=0,
    )
    background = np.median(border_pixels, axis=0)
    diff = np.linalg.norm(arr - background, axis=2)
    foreground_mask = _dilate_mask(diff > max(float(np.percentile(diff, 74)), 20.0), radius=1)

    boxes = _sanitize_boxes(_connected_components(foreground_mask), working_width, working_height)
    boxes = _sanitize_boxes(_merge_close_boxes(boxes, gap=max(4, working_width // 80)), working_width, working_height)
    boxes = sorted(boxes, key=lambda box: box[0])[:max_items]

    if len(boxes) <= 1:
        return []

    regions = []
    for idx, (left_small, top_small, right_small, bottom_small, _) in enumerate(boxes, start=1):
        pad_x = max(int((right_small - left_small) * 0.12), 8)
        pad_y = max(int((bottom_small - top_small) * 0.12), 8)
        left_small = max(left_small - pad_x, 0)
        right_small = min(right_small + pad_x, working_width)
        top_small = max(top_small - pad_y, 0)
        bottom_small = min(bottom_small + pad_y, working_height)
        left = int(left_small * original_width / working_width)
        right = int(right_small * original_width / working_width)
        top = int(top_small * original_height / working_height)
        bottom = int(bottom_small * original_height / working_height)
        regions.append((idx, rgb.crop((left, top, right, bottom))))
    return regions


def grade_item(confidence: float, is_low_confidence: bool):
    if is_low_confidence:
        return "Uncertain", "Please inspect manually before consuming.", "warning"
    if confidence >= 85:
        return "Likely edible", "Strong visual candidate for consumption.", "success"
    if confidence >= 60:
        return "Possibly edible", "Looks acceptable, but inspect for damage first.", "info"
    return "Weak signal", "Model confidence is too low to recommend confidently.", "warning"


def render_status_box(kind: str, text: str):
    if kind == "success":
        st.success(text)
    elif kind == "info":
        st.info(text)
    else:
        st.warning(text)


st.markdown(
    """
    <style>
    :root {
        --bg-a: #f6f0e4;
        --bg-b: #eef7f0;
        --panel: rgba(255, 255, 255, 0.90);
        --panel-strong: #ffffff;
        --text: #17261f;
        --muted: #62756b;
        --line: rgba(23, 63, 46, 0.10);
        --shadow: 0 18px 44px rgba(24, 54, 41, 0.08);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.75), transparent 30%),
            linear-gradient(180deg, var(--bg-a), var(--bg-b));
        color: var(--text);
    }
    .block-container {
        max-width: 1220px;
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
    }
    .hero {
        background: linear-gradient(135deg, rgba(27, 92, 61, 0.98), rgba(66, 123, 80, 0.94));
        color: white;
        border-radius: 28px;
        padding: 30px;
        margin-bottom: 18px;
        box-shadow: var(--shadow);
    }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
        opacity: 0.82;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 2.4rem;
        line-height: 1.02;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .hero-copy {
        max-width: 760px;
        color: rgba(255,255,255,0.86);
        font-size: 1rem;
    }
    .card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
    }
    .section-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 6px;
    }
    .section-copy {
        color: var(--muted);
        font-size: 0.94rem;
        margin-bottom: 12px;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
    }
    .stat {
        background: var(--panel-strong);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 12px 14px;
    }
    .stat-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
         font-size: 0.75rem;
        margin-bottom: 4px;
    }
    .stat-value {
        color: var(--text);
        font-weight: 700;
        font-size: 1.02rem;
        word-break: break-word;
    }
    [data-testid="stRadio"] label p {
        font-weight: 600;
        color: var(--text) !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 999px;
        padding: 6px 12px;
        border: 1px solid var(--line);
    }
    [data-testid="stAlert"] {
        color: #17261f !important;
    }
    [data-testid="stAlert"] * {
        color: #17261f !important;
    }
    [data-testid="stCaptionContainer"] {
        color: #24362e !important;
    }
    [data-testid="stCaptionContainer"] * {
        color: #24362e !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Smart produce analysis</div>
        <div class="hero-title">Single-image fruit and vegetable quality guidance.</div>
        <div class="hero-copy">
            Upload one clear fruit or vegetable photo and the model will classify it and give
            a confidence-based edibility guidance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

controls_col, help_col = st.columns([1.45, 1], gap="large")

with controls_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Input Setup</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Use one fruit or vegetable per image for reliable results.</div>",
        unsafe_allow_html=True,
    )
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
    else:
        camera_file = st.camera_input("Capture Image")
        if camera_file:
            image = Image.open(camera_file)
    enable_multi_scan = st.checkbox("Enable beta multi-item scan", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

with help_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Best Results</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Keep one item centered, use even lighting, avoid cluttered backgrounds, "
        "and make the fruit or vegetable fill most of the frame.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


if image is not None:
    try:
        predicted_class, confidence, is_low_confidence, preprocessing_mode, model_name = predict_image(image)
    except Exception as e:
        status = get_model_status()
        st.error("Prediction failed because the model could not be loaded.")
        st.error(str(e))
        if status["errors"]:
            st.caption("Load errors:")
            for err in status["errors"]:
                st.write(f"- {err}")
        st.stop()

    preview_col, result_col = st.columns([1.05, 1.35], gap="large")

    with preview_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Image Preview</div>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Source</div>
                    <div class="stat-value">{option}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Canvas</div>
                    <div class="stat-value">{image.size[0]} x {image.size[1]}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        st.markdown(f"### Identified as: **{predicted_class.capitalize()}**")
        st.progress(int(confidence))
        st.markdown(
            f"""
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Confidence</div>
                    <div class="stat-value">{confidence:.2f}%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Preprocess</div>
                    <div class="stat-value">{preprocessing_mode}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Model</div>
                    <div class="stat-value">{model_name}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Mode</div>
                    <div class="stat-value">{
                        'Standard classification + beta multi-item scan'
                        if enable_multi_scan else 'Standard classification'
                    }</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Edibility Guidance</div>", unsafe_allow_html=True)
        label, detail, kind = grade_item(confidence, is_low_confidence)
        render_status_box(kind, f"{label}: {detail}")
        st.markdown("</div>", unsafe_allow_html=True)

    if enable_multi_scan:
        multi_regions = detect_item_regions(image, max_items=6)
        if len(multi_regions) > 1:
            results = []
            for idx, region in multi_regions:
                item_class, item_confidence, item_low_confidence, _, _ = predict_image(region)
                results.append(
                    {
                        "index": idx,
                        "region": region,
                        "item_class": item_class,
                        "confidence": item_confidence,
                        "low_confidence": item_low_confidence,
                    }
                )

            best_item = sorted(results, key=lambda item: item["confidence"], reverse=True)[0]

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Beta Multi-Item Comparison</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="compare-banner">
                    <div class="compare-title">Best candidate</div>
                    <div class="compare-copy">
                        Item {best_item['index']} looks most edible in this image
                        ({best_item['item_class']}, {best_item['confidence']:.2f}% confidence).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            columns_per_row = 3 if len(results) <= 6 else 4
            for row_start in range(0, len(results), columns_per_row):
                row_items = results[row_start:row_start + columns_per_row]
                row_columns = st.columns(len(row_items), gap="medium")
                for item, column in zip(row_items, row_columns):
                    with column:
                        st.image(item["region"], caption=f"Item {item['index']}", use_container_width=True)
                        st.markdown(f"**{item['item_class'].capitalize()}**")
                        st.write(f"{item['confidence']:.2f}% confidence")
                        if item["index"] == best_item["index"] and not item["low_confidence"]:
                            st.success("Best edible choice")
                        else:
                            label, detail, kind = grade_item(item["confidence"], item["low_confidence"])
                            render_status_box(kind, f"{label}: {detail}")

            st.caption("Beta mode uses heuristic region detection and may be wrong on crowded images.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Beta multi-item scan did not find clearly separate items in this image.")

    st.caption("AI Model: Optimized GoogLeNet (InceptionV3) | Built using TensorFlow & Streamlit")
