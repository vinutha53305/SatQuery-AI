import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

from ai_model import SatelliteAI
from caption_model import SatelliteCaptioner
from land_cover import LandCoverAnalyzer


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
)


st.markdown(
    """
    <style>

    .stApp {
        background: #07131d;
        color: white;
    }

    [data-testid="stSidebar"] {
        background: #0a1b27;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0c2636, #0a1824);
        border: 1px solid #173b50;
        margin-bottom: 25px;
    }

    .badge {
        color: #55c7e8;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin-top: 8px;
    }

    .hero-subtitle {
        color: #91a9b8;
        font-size: 17px;
        margin-top: 8px;
    }

    .card {
        background: #0c202e;
        border: 1px solid #17394c;
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0;
    }

    .metric-card {
        background: #0c202e;
        border: 1px solid #17394c;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #60d5f5;
    }

    .metric-label {
        color: #8fa9bb;
        margin-top: 5px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .workflow {
        background: #0c202e;
        border: 1px solid #17394c;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
    }

    .workflow-number {
        font-size: 26px;
        font-weight: 800;
        color: #60d5f5;
    }

    .workflow-title {
        font-size: 17px;
        font-weight: 700;
        margin-top: 8px;
    }

    .workflow-text {
        color: #8fa9bb;
        font-size: 14px;
        margin-top: 5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


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
    return Image.open(uploaded_file).convert("RGB")


def image_download_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def route_question(question):

    question = question.lower()

    change_words = [
        "change",
        "changed",
        "difference",
        "different",
        "compare",
        "comparison",
        "before",
        "after",
        "construction",
        "demolition",
        "growth",
        "development"
    ]

    caption_words = [
        "describe",
        "description",
        "caption",
        "scene",
        "summarize",
        "summary",
        "what is in this image"
    ]

    land_cover_words = [
        "land cover",
        "land-cover",
        "urban",
        "built up",
        "vegetation",
        "forest",
        "water",
        "agriculture",
        "agricultural",
        "road",
        "roads"
    ]

    if any(word in question for word in change_words):
        return "change"

    if any(word in question for word in land_cover_words):
        return "land_cover"

    if any(word in question for word in caption_words):
        return "caption"

    return "vqa"


def analyze_change(before, after, threshold):

    width = min(before.width, after.width)
    height = min(before.height, after.height)

    before = before.resize((width, height))
    after = after.resize((width, height))

    before_array = np.array(before).astype(np.int16)
    after_array = np.array(after).astype(np.int16)

    difference = np.abs(before_array - after_array)

    difference_gray = np.max(difference, axis=2)

    mask = difference_gray > threshold

    changed_pixels = int(np.sum(mask))
    total_pixels = int(mask.size)

    if total_pixels > 0:
        change_percentage = (
            changed_pixels / total_pixels
        ) * 100
    else:
        change_percentage = 0

    difference_image = Image.fromarray(
        np.clip(
            difference_gray,
            0,
            255
        ).astype(np.uint8)
    )

    mask_image = Image.fromarray(
        mask.astype(np.uint8) * 255
    )

    overlay_array = np.array(before).copy()

    overlay_array[mask] = [255, 0, 0]

    overlay_image = Image.fromarray(
        overlay_array
    )

    return (
        before,
        after,
        difference_image,
        mask_image,
        overlay_image,
        changed_pixels,
        total_pixels,
        change_percentage
    )


def get_change_level(change_percentage):

    if change_percentage < 1:
        return "Minimal"

    if change_percentage < 5:
        return "Low"

    if change_percentage < 15:
        return "Moderate"

    return "High"


st.sidebar.markdown(
    """
    <div style="
        font-size:25px;
        font-weight:800;
        color:#60d5f5;
        padding:10px 0 20px 0;
    ">
    🛰️ SatQuery AI
    </div>
    """,
    unsafe_allow_html=True
)


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

    st.markdown(
        """
        <div class="hero">
        <div class="badge">SATELLITE INTELLIGENCE PLATFORM</div>
        <div class="hero-title">
        🛰️ SatQuery AI
        </div>
        <div class="hero-subtitle">
        Interactive Vision-Language Assistant for Satellite Image Analysis
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🚀 Platform Capabilities</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="card">
            <h3>🤖 AI Agent</h3>
            <p>
            Ask natural-language questions about satellite images
            and receive AI-generated answers.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
            <h3>🛰️ Change Analysis</h3>
            <p>
            Compare satellite images from different dates
            and visualize potential changes.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="card">
            <h3>🌍 Land Cover</h3>
            <p>
            Estimate possible land-cover categories using
            a vision-language model.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">⚙️ AI Workflow</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    workflow = [
        ("01", "Upload", "Provide satellite imagery"),
        ("02", "Query", "Ask a natural-language question"),
        ("03", "Analyze", "AI processes the imagery"),
        ("04", "Respond", "Receive an intelligent result")
    ]

    for col, item in zip(
        [c1, c2, c3, c4],
        workflow
    ):

        with col:

            st.markdown(
                f"""
                <div class="workflow">

                <div class="workflow-number">
                {item[0]}
                </div>

                <div class="workflow-title">
                {item[1]}
                </div>

                <div class="workflow-text">
                {item[2]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


elif page == "🤖 AI Agent":

    st.markdown(
        """
        <div class="hero">
        <div class="badge">VISION-LANGUAGE INTELLIGENCE</div>
        <div class="hero-title">
        🤖 AI Agent
        </div>
        <div class="hero-subtitle">
        Ask questions about your satellite image using natural language.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "🛰️ Upload a satellite image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff"
        ],
        key="agent_image"
    )

    if uploaded_file is not None:

        image = load_image(uploaded_file)

        st.image(
            image,
            caption="Uploaded Satellite Image",
            use_container_width=True
        )

        question = st.text_input(
            "💬 Ask your question",
            placeholder="Example: What is visible in this image?"
        )

        if st.button(
            "🤖 Analyze",
            use_container_width=True
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                route = route_question(question)

                if route == "change":

                    st.warning(
                        "🛰️ This question requires comparing "
                        "two satellite images. Please use the "
                        "Change Analysis section."
                    )

                elif route == "caption":

                    with st.spinner(
                        "Generating image description..."
                    ):

                        model = get_caption_model()

                        answer = model.generate_caption(
                            image
                        )

                    st.subheader("🧠 AI Response")

                    st.markdown(
                        f"""
                        <div class="card">
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif route == "land_cover":

                    with st.spinner(
                        "Analyzing land-cover patterns..."
                    ):

                        model = get_land_cover_model()

                        results = model.analyze(
                            image
                        )

                    st.subheader(
                        "🌍 Land-Cover Estimation"
                    )

                    for result in results:

                        label = result["label"]
                        confidence = result["confidence"]

                        if confidence >= 20:

                            st.markdown(
                                f"""
                                <div class="card">

                                <div style="
                                    display:flex;
                                    justify-content:space-between;
                                    align-items:center;
                                ">

                                <div>
                                <div style="
                                    font-size:19px;
                                    font-weight:700;
                                    color:white;
                                ">
                                🌍 {label}
                                </div>

                                <div style="
                                    color:#8fa9bb;
                                    margin-top:5px;
                                ">
                                AI estimated confidence
                                </div>
                                </div>

                                <div style="
                                    font-size:23px;
                                    font-weight:800;
                                    color:#60d5f5;
                                ">
                                {confidence:.1f}%
                                </div>

                                </div>

                                <div style="
                                    background:#142838;
                                    border-radius:10px;
                                    height:10px;
                                    margin-top:15px;
                                    overflow:hidden;
                                ">

                                <div style="
                                    width:{min(confidence,100):.1f}%;
                                    background:linear-gradient(
                                        90deg,
                                        #176b87,
                                        #32b5d3
                                    );
                                    height:100%;
                                    border-radius:10px;
                                ">
                                </div>

                                </div>

                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                else:

                    with st.spinner(
                        "AI is analyzing the satellite image..."
                    ):

                        model = get_vqa_model()

                        answer = model.answer_question(
                            image,
                            question
                        )

                    st.subheader("🧠 AI Response")

                    st.markdown(
                        f"""
                        <div class="card">
                        {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


elif page == "🛰️ Change Analysis":

    st.markdown(
        """
        <div class="hero">
        <div class="badge">BI-TEMPORAL SATELLITE ANALYSIS</div>
        <div class="hero-title">
        🛰️ Change Analysis
        </div>
        <div class="hero-subtitle">
        Compare two satellite images and visualize potential changes.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        before_file = st.file_uploader(
            "📅 Before Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff"
            ],
            key="before_image"
        )

    with col2:

        after_file = st.file_uploader(
            "📅 After Image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff"
            ],
            key="after_image"
        )

    threshold = st.slider(
        "🎚️ Pixel Difference Threshold",
        min_value=5,
        max_value=100,
        value=30
    )

    if before_file is not None and after_file is not None:

        before_image = load_image(
            before_file
        )

        after_image = load_image(
            after_file
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                before_image,
                caption="Before",
                use_container_width=True
            )

        with col2:

            st.image(
                after_image,
                caption="After",
                use_container_width=True
            )

        if st.button(
            "🛰️ Analyze Changes",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing satellite images..."
            ):

                (
                    before,
                    after,
                    difference_image,
                    mask_image,
                    overlay_image,
                    changed_pixels,
                    total_pixels,
                    change_percentage
                ) = analyze_change(
                    before_image,
                    after_image,
                    threshold
                )

            change_level = get_change_level(
                change_percentage
            )

            st.subheader(
                "📊 Change Analysis Results"
            )

            m1, m2, m3 = st.columns(3)

            with m1:

                st.markdown(
                    f"""
                    <div class="metric-card">
                    <div class="metric-value">
                    {changed_pixels:,}
                    </div>
                    <div class="metric-label">
                    Changed Pixels
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m2:

                st.markdown(
                    f"""
                    <div class="metric-card">
                    <div class="metric-value">
                    {change_percentage:.2f}%
                    </div>
                    <div class="metric-label">
                    Potential Change
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with m3:

                st.markdown(
                    f"""
                    <div class="metric-card">
                    <div class="metric-value">
                    {change_level}
                    </div>
                    <div class="metric-label">
                    Change Level
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="section-title">🔎 Visual Analysis</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.image(
                    difference_image,
                    caption="Difference Image",
                    use_container_width=True
                )

            with c2:

                st.image(
                    mask_image,
                    caption="Change Mask",
                    use_container_width=True
                )

            with c3:

                st.image(
                    overlay_image,
                    caption="Potential Changes",
                    use_container_width=True
                )

            st.download_button(
                "⬇️ Download Change Overlay",
                data=image_download_bytes(
                    overlay_image
                ),
                file_name="satquery_change_overlay.png",
                mime="image/png",
                use_container_width=True
            )

            st.warning(
                "⚠️ Potential changes can also be caused by "
                "clouds, shadows, lighting, seasonal differences, "
                "image misalignment, or sensor differences. "
                "Results should be reviewed before making decisions."
            )

    else:

        st.info(
            "Upload both Before and After satellite images "
            "to perform change analysis."
        )


elif page == "🌍 Land Cover":

    st.markdown(
        """
        <div class="hero">
        <div class="badge">AI LAND-COVER ESTIMATION</div>
        <div class="hero-title">
        🌍 Land-Cover Analysis
        </div>
        <div class="hero-subtitle">
        Estimate possible land-cover categories from a satellite image.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "🛰️ Upload a satellite image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff"
        ],
        key="land_cover_image"
    )

    if uploaded_file is not None:

        image = load_image(
            uploaded_file
        )

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
                "AI is analyzing land-cover patterns..."
            ):

                model = get_land_cover_model()

                results = model.analyze(
                    image
                )

            st.subheader(
                "🔎 Estimated Land-Cover Types"
            )

            detected = [
                result
                for result in results
                if result["confidence"] >= 20
            ]

            if detected:

                for result in detected:

                    label = result["label"]
                    confidence = result["confidence"]

                    st.markdown(
                        f"""
                        <div class="card">

                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:center;
                        ">

                        <div>

                        <div style="
                            font-size:20px;
                            font-weight:700;
                            color:white;
                        ">
                        🌍 {label}
                        </div>

                        <div style="
                            color:#8fa9bb;
                            margin-top:5px;
                        ">
                        AI estimated confidence
                        </div>

                        </div>

                        <div style="
                            font-size:24px;
                            font-weight:800;
                            color:#60d5f5;
                        ">
                        {confidence:.1f}%
                        </div>

                        </div>

                        <div style="
                            background:#142838;
                            border-radius:10px;
                            height:10px;
                            margin-top:15px;
                            overflow:hidden;
                        ">

                        <div style="
                            width:{min(confidence,100):.1f}%;
                            background:linear-gradient(
                                90deg,
                                #176b87,
                                #32b5d3
                            );
                            height:100%;
                            border-radius:10px;
                        ">
                        </div>

                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.info(
                    "No land-cover category reached "
                    "the confidence threshold."
                )

            st.markdown("---")

            st.subheader(
                "📊 All Category Scores"
            )

            for result in results:

                label = result["label"]
                confidence = result["confidence"]

                st.write(
                    f"**{label}** — {confidence:.1f}%"
                )

            st.markdown("---")

            st.warning(
                "⚠️ These confidence scores are AI estimates "
                "from a general-purpose vision-language model. "
                "They are intended for demonstration and "
                "decision-support purposes, not scientifically "
                "validated land-cover mapping."
            )

    else:

        st.info(
            "Upload a satellite image to perform land-cover analysis."
        )


st.markdown(
    """
    <br><br>

    <div style="
        text-align:center;
        color:#607d8b;
        padding:20px;
        border-top:1px solid #17394c;
    ">

    🛰️ <b>SatQuery AI</b><br>

    Interactive Vision-Language Assistant for Satellite Image Analysis<br>

    <span style="font-size:13px;">
    SIH 2026 Prototype
    </span>

    </div>
    """,
    unsafe_allow_html=True
)