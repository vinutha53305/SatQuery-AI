import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

from app.ai_model import SatelliteAI
from app.caption_model import SatelliteCaptioner
from app.land_cover import LandCoverAnalyzer


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background: #07111f;
    color: #e8f1ff;
}

section[data-testid="stSidebar"] {
    background: #0b1728;
}

h1, h2, h3 {
    color: #dcecff;
}

.card {
    background: #0d1b2e;
    border: 1px solid #1d3557;
    border-radius: 15px;
    padding: 22px;
    margin-bottom: 18px;
}

.metric-card {
    background: #10233a;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    border: 1px solid #21476d;
}

.small-text {
    color: #9fb3c8;
}

.answer-box {
    background: #0d1b2e;
    border-left: 4px solid #4da3ff;
    padding: 20px;
    border-radius: 10px;
    margin-top: 15px;
}

.warning-box {
    background: #2a2110;
    border-left: 4px solid #e5a93d;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vqa_model():
    return SatelliteAI()


@st.cache_resource
def get_caption_model():
    return SatelliteCaptioner()


@st.cache_resource
def get_land_cover_model():
    return LandCoverAnalyzer()


def load_image(uploaded_file):
    if uploaded_file is None:
        return None

    return Image.open(uploaded_file).convert("RGB")


def image_download_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def route_question(question):

    q = question.lower().strip()

    change_words = [
        "change",
        "changed",
        "difference",
        "different",
        "compare",
        "comparison",
        "before and after",
        "before vs after",
        "construction",
        "demolition",
        "growth",
        "development"
    ]

    land_cover_words = [
        "land cover",
        "land-cover",
        "land type",
        "urban",
        "built-up",
        "built up",
        "vegetation",
        "forest",
        "water body",
        "water bodies",
        "agriculture",
        "agricultural",
        "farmland",
        "road",
        "roads"
    ]

    caption_words = [
        "describe",
        "description",
        "caption",
        "scene",
        "summarize",
        "summary",
        "what is in this image",
        "what can you see",
        "what do you see",
        "tell me about this image"
    ]

    full_analysis_words = [
        "explain everything",
        "everything about this image",
        "analyze this image",
        "analyse this image",
        "complete analysis",
        "full analysis",
        "detailed analysis",
        "detailed description",
        "complete description",
        "analyze everything",
        "analyse everything",
        "give me details",
        "give details",
        "tell me everything",
        "overall analysis",
        "overall description",
        "what is visible"
    ]

    for word in change_words:
        if word in q:
            return "change"

    for word in full_analysis_words:
        if word in q:
            return "full_analysis"

    for word in land_cover_words:
        if word in q:
            return "land_cover"

    for word in caption_words:
        if word in q:
            return "caption"

    return "vqa"


def get_change_level(percent):

    if percent < 5:
        return "Low"
    elif percent < 15:
        return "Moderate"
    elif percent < 30:
        return "High"
    else:
        return "Very High"


def analyze_change(before, after, threshold):

    before_array = np.array(before).astype(np.int16)
    after_array = np.array(after).astype(np.int16)

    min_height = min(
        before_array.shape[0],
        after_array.shape[0]
    )

    min_width = min(
        before_array.shape[1],
        after_array.shape[1]
    )

    before_array = before_array[:min_height, :min_width]
    after_array = after_array[:min_height, :min_width]

    difference = np.abs(
        before_array - after_array
    )

    difference_score = difference.mean(axis=2)

    mask = difference_score > threshold

    changed_pixels = int(mask.sum())
    total_pixels = int(mask.size)

    change_percentage = (
        changed_pixels / total_pixels
    ) * 100

    diff_image = np.clip(
        difference,
        0,
        255
    ).astype(np.uint8)

    mask_image = (
        mask.astype(np.uint8) * 255
    )

    overlay = after_array.copy()

    overlay[mask] = [
        255,
        0,
        0
    ]

    return (
        diff_image,
        mask_image,
        overlay,
        changed_pixels,
        total_pixels,
        change_percentage
    )


def generate_full_analysis(image):

    caption_model = get_caption_model()
    land_model = get_land_cover_model()

    caption = caption_model.generate_caption(image)

    land_results = land_model.analyze(image)

    significant_results = [
        result
        for result in land_results
        if result["confidence"] >= 20
    ]

    if not significant_results:
        significant_results = land_results[:3]

    return caption, significant_results
st.sidebar.title("🛰️ SatQuery AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🤖 AI Agent",
        "🛰️ Change Analysis",
        "🌍 Land Cover"
    ]
)


if page == "🏠 Dashboard":

    st.title("🛰️ SatQuery AI")

    st.markdown(
        "### Interactive Vision-Language Assistant for Satellite Image Analysis"
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🤖 AI Agent</h3>
        <p class="small-text">
        Ask natural-language questions about satellite images.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🛰️ Change Analysis</h3>
        <p class="small-text">
        Compare satellite images from different dates.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
        <h3>🌍 Land Cover</h3>
        <p class="small-text">
        Estimate major land-cover categories.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🔄 AI Workflow")

    workflow = st.columns(4)

    with workflow[0]:
        st.markdown("""
        <div class="metric-card">
        <h3>1️⃣</h3>
        <b>Upload</b>
        <p class="small-text">Satellite image</p>
        </div>
        """, unsafe_allow_html=True)

    with workflow[1]:
        st.markdown("""
        <div class="metric-card">
        <h3>2️⃣</h3>
        <b>Query</b>
        <p class="small-text">Natural language</p>
        </div>
        """, unsafe_allow_html=True)

    with workflow[2]:
        st.markdown("""
        <div class="metric-card">
        <h3>3️⃣</h3>
        <b>Analyze</b>
        <p class="small-text">AI models</p>
        </div>
        """, unsafe_allow_html=True)

    with workflow[3]:
        st.markdown("""
        <div class="metric-card">
        <h3>4️⃣</h3>
        <b>Respond</b>
        <p class="small-text">Actionable result</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.info(
        "SatQuery AI combines vision-language analysis, "
        "land-cover estimation and image-change detection "
        "into a single interactive platform."
    )


elif page == "🤖 AI Agent":

    st.title("🤖 AI Agent")

    st.write(
        "Upload a satellite image and ask a natural-language question."
    )

    uploaded_file = st.file_uploader(
        "📡 Upload Satellite Image",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

    if uploaded_file:

        image = load_image(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption="Uploaded Satellite Image",
                use_container_width=True
            )

        with col2:

            st.markdown("### 💬 Ask the AI Agent")

            question = st.text_input(
                "Your question",
                placeholder="Example: Explain everything about this image"
            )

            analyze_button = st.button(
                "🚀 Analyze Image",
                use_container_width=True
            )

        if analyze_button:

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                route = route_question(question)

                st.markdown("---")

                if route == "change":

                    st.warning(
                        "🛰️ Change analysis requires two satellite "
                        "images from different dates."
                    )

                    st.info(
                        "Go to **🛰️ Change Analysis** from the sidebar "
                        "to upload Before and After images."
                    )

                elif route == "full_analysis":

                    with st.spinner(
                        "🧠 Performing complete satellite image analysis..."
                    ):

                        caption, land_results = generate_full_analysis(
                            image
                        )

                    st.markdown("## 🧠 Satellite Image Analysis")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        <h3>📡 Overall Description</h3>
                        <p>{caption}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("### 🌍 Estimated Land-Cover Categories")

                    cols = st.columns(
                        min(len(land_results), 3)
                    )

                    for index, result in enumerate(land_results):

                        with cols[index % len(cols)]:

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                <h3>{result["label"]}</h3>
                                <h2>{result["confidence"]:.1f}%</h2>
                                <p class="small-text">
                                AI estimated score
                                </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.markdown("### 🔎 Interpretation")

                    top_category = land_results[0]["label"]

                    st.write(
                        f"The strongest estimated category in this "
                        f"image is **{top_category}** based on the "
                        f"vision-language model's comparison of the "
                        f"candidate land-cover descriptions."
                    )

                    st.markdown("""
                    <div class="warning-box">
                    ⚠️ <b>Important:</b> These land-cover scores are
                    AI-assisted estimates from a general-purpose
                    vision-language model. They are not a replacement
                    for scientifically validated remote-sensing
                    classification.
                    </div>
                    """, unsafe_allow_html=True)

                elif route == "caption":

                    with st.spinner(
                        "📝 Generating image description..."
                    ):

                        caption = get_caption_model().generate_caption(
                            image
                        )

                    st.markdown("## 📝 Image Description")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        <p>{caption}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif route == "land_cover":

                    with st.spinner(
                        "🌍 Analyzing land cover..."
                    ):

                        results = get_land_cover_model().analyze(
                            image
                        )

                    st.markdown("## 🌍 Land-Cover Analysis")

                    cols = st.columns(3)

                    for index, result in enumerate(results):

                        with cols[index % 3]:

                            st.markdown(
                                f"""
                                <div class="metric-card">
                                <h3>{result["label"]}</h3>
                                <h2>{result["confidence"]:.1f}%</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    st.warning(
                        "These are AI-assisted relative scores, "
                        "not scientifically validated land-cover "
                        "classification probabilities."
                    )

                else:

                    with st.spinner(
                        "🤖 AI is analyzing your question..."
                    ):

                        answer = get_vqa_model().answer_question(
                            image,
                            question
                        )

                    st.markdown("## 🤖 AI Response")

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        <p>{answer}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


elif page == "🛰️ Change Analysis":

    st.title("🛰️ Bi-Temporal Change Analysis")

    st.write(
        "Compare two satellite images of the same area "
        "from different dates."
    )

    col1, col2 = st.columns(2)

    with col1:

        before_file = st.file_uploader(
            "📅 Before Image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="before"
        )

    with col2:

        after_file = st.file_uploader(
            "📅 After Image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="after"
        )

    threshold = st.slider(
        "🎚️ Change Detection Threshold",
        min_value=5,
        max_value=100,
        value=30
    )

    if before_file and after_file:

        before = load_image(before_file)
        after = load_image(after_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                before,
                caption="Before",
                use_container_width=True
            )

        with col2:
            st.image(
                after,
                caption="After",
                use_container_width=True
            )

        if st.button(
            "🔍 Detect Changes",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image differences..."
            ):

                (
                    diff_image,
                    mask_image,
                    overlay,
                    changed_pixels,
                    total_pixels,
                    change_percentage
                ) = analyze_change(
                    before,
                    after,
                    threshold
                )

            st.markdown("---")

            st.markdown("## 📊 Change Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Changed Pixels",
                    f"{changed_pixels:,}"
                )

            with col2:
                st.metric(
                    "Total Pixels",
                    f"{total_pixels:,}"
                )

            with col3:
                st.metric(
                    "Potential Change",
                    f"{change_percentage:.2f}%"
                )

            level = get_change_level(
                change_percentage
            )

            st.info(
                f"Detected change level: **{level}**"
            )

            st.markdown("## 🛰️ Change Visualization")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    diff_image,
                    caption="Difference Image",
                    use_container_width=True
                )

            with col2:
                st.image(
                    mask_image,
                    caption="Binary Change Mask",
                    use_container_width=True
                )

            with col3:
                st.image(
                    overlay,
                    caption="Red Change Overlay",
                    use_container_width=True
                )

            st.download_button(
                "⬇️ Download Change Overlay",
                data=image_download_bytes(
                    Image.fromarray(overlay)
                ),
                file_name="satquery_change_overlay.png",
                mime="image/png"
            )

            st.markdown("### ⚠️ Interpretation Warning")

            st.warning(
                "Pixel differences do not automatically mean real-world "
                "land-use change. Clouds, shadows, seasonal variation, "
                "lighting differences, image misalignment and sensor "
                "differences can produce false changes."
            )


elif page == "🌍 Land Cover":

    st.title("🌍 Land-Cover Analysis")

    st.write(
        "Estimate major land-cover categories using "
        "a vision-language model."
    )

    uploaded_file = st.file_uploader(
        "📡 Upload Satellite Image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="landcover"
    )

    if uploaded_file:

        image = load_image(uploaded_file)

        st.image(
            image,
            caption="Satellite Image",
            use_container_width=True
        )

        if st.button(
            "🌍 Analyze Land Cover",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing land-cover categories..."
            ):

                results = get_land_cover_model().analyze(
                    image
                )

            st.markdown("---")

            st.markdown("## 🌍 Estimated Categories")

            cols = st.columns(3)

            for index, result in enumerate(results):

                with cols[index % 3]:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                        <h3>{result["label"]}</h3>
                        <h2>{result["confidence"]:.1f}%</h2>
                        <p class="small-text">
                        AI estimated score
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("### 📊 Model Scores")

            for result in results:

                st.write(
                    f"**{result['label']}** — "
                    f"{result['confidence']:.2f}%"
                )

                st.progress(
                    min(
                        int(result["confidence"]),
                        100
                    )
                )

            st.markdown("""
            <div class="warning-box">
            ⚠️ These scores are generated by a general-purpose
            vision-language model and should be treated as
            AI-assisted estimates rather than scientifically
            validated remote-sensing classification.
            </div>
            """, unsafe_allow_html=True)


st.markdown("---")

st.markdown(
    "<center>🛰️ SatQuery AI • SIH 2026 Prototype</center>",
    unsafe_allow_html=True
)