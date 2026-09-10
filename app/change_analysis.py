import streamlit as st
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CHANGE ANALYSIS
# ============================================================

class ChangeAnalyzer:

    def compare(self, before_image, after_image):

        # Convert to RGB
        before = before_image.convert("RGB")
        after = after_image.convert("RGB")

        # Resize to common dimensions
        width = min(before.width, after.width)
        height = min(before.height, after.height)

        before = before.resize((width, height))
        after = after.resize((width, height))

        # Convert to NumPy arrays
        before_array = np.asarray(before).astype(np.int16)
        after_array = np.asarray(after).astype(np.int16)

        # Absolute pixel difference
        difference_array = np.abs(
            before_array - after_array
        )

        # Mean difference across RGB channels
        difference_gray = difference_array.mean(axis=2)

        # Threshold
        threshold = 30

        change_mask_array = (
            difference_gray > threshold
        ).astype(np.uint8) * 255

        # Count changed pixels
        changed_pixels = int(
            np.count_nonzero(change_mask_array)
        )

        total_pixels = width * height

        change_percentage = (
            changed_pixels / total_pixels * 100
            if total_pixels > 0
            else 0
        )

        # Create difference image
        difference_display = np.clip(
            difference_array,
            0,
            255
        ).astype(np.uint8)

        difference_image = Image.fromarray(
            difference_display
        )

        # Enhance difference visualization
        difference_image = ImageEnhance.Contrast(
            difference_image
        ).enhance(3.0)

        # Change mask image
        change_mask = Image.fromarray(
            change_mask_array
        )

        # Create red overlay on AFTER image
        after_array_uint8 = np.asarray(after).copy()

        # Highlight detected changes
        after_array_uint8[change_mask_array > 0] = [
            255,
            0,
            0
        ]

        change_overlay = Image.fromarray(
            after_array_uint8
        )

        return {
            "before": before,
            "after": after,
            "difference": difference_image,
            "change_mask": change_mask,
            "change_overlay": change_overlay,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": change_percentage
        }


# ============================================================
# HEADER
# ============================================================

st.title("🛰️ SatQuery AI")

st.subheader(
    "Interactive Vision-Language Assistant "
    "for Satellite Image Analysis"
)

st.write(
    "Analyze satellite imagery using AI-assisted "
    "image understanding and bi-temporal change detection."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧭 Navigation")

    page = st.radio(
        "Choose a module",
        [
            "🏠 Home",
            "📡 Single Image Analysis",
            "🛰️ Change Analysis",
            "🤖 AI Agent"
        ]
    )

    st.divider()

    st.caption(
        "SatQuery AI — SIH 2026 Prototype"
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header("Welcome to SatQuery AI 👋")

    st.write(
        """
        SatQuery AI is an interactive satellite image
        analysis prototype designed to help users understand
        Earth observation imagery using AI.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Single Image VQA",
            "Active"
        )

    with col2:
        st.metric(
            "Scene Analysis",
            "Active"
        )

    with col3:
        st.metric(
            "Change Detection",
            "Active"
        )

    st.divider()

    st.subheader("🚀 Project Workflow")

    st.markdown(
        """
        **1. Upload satellite image**

        ↓

        **2. AI-based image understanding**

        ↓

        **3. Ask questions / generate analysis**

        ↓

        **4. Upload before & after images**

        ↓

        **5. Detect potential changes**

        ↓

        **6. Visualize the change map**
        """
    )

    st.info(
        "This is a prototype. Pixel differences can be "
        "caused by lighting, clouds, shadows, seasonal "
        "changes, or image misalignment."
    )


# ============================================================
# SINGLE IMAGE ANALYSIS
# ============================================================

elif page == "📡 Single Image Analysis":

    st.header("📡 Upload Satellite Image")

    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "tif",
            "tiff"
        ],
        key="single_image"
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.success(
                "Image uploaded successfully!"
            )

            st.image(
                image,
                caption="Uploaded Satellite Image",
                use_container_width=True
            )

            st.write(
                f"**Image size:** "
                f"{image.width} × {image.height} pixels"
            )

            st.write(
                f"**Format:** {uploaded_file.type}"
            )

            st.divider()

            st.subheader("🤖 Image Analysis")

            question = st.text_input(
                "Ask a question about the image",
                placeholder=(
                    "Example: What can you see "
                    "in this satellite image?"
                )
            )

            if st.button(
                "🔍 Analyze Image",
                use_container_width=True
            ):

                if question.strip():

                    st.info(
                        "AI question-answering module "
                        "is ready to be connected to "
                        "your vision-language model."
                    )

                    st.write(
                        "**Your question:**"
                    )

                    st.write(question)

                else:

                    st.warning(
                        "Please enter a question."
                    )

        except Exception as e:

            st.error(
                f"Unable to process image: {e}"
            )


# ============================================================
# BI-TEMPORAL CHANGE ANALYSIS
# ============================================================

elif page == "🛰️ Change Analysis":

    st.header(
        "🛰️ Bi-Temporal Change Analysis"
    )

    st.write(
        "Upload two images of the same area from "
        "different dates to visualize potential changes."
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    with col1:

        st.subheader("📅 Before")

        before_file = st.file_uploader(
            "Upload the earlier image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff"
            ],
            key="before"
        )

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    with col2:

        st.subheader("📅 After")

        after_file = st.file_uploader(
            "Upload the later image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff"
            ],
            key="after"
        )

    before_image = None
    after_image = None

    if before_file is not None:

        try:

            before_image = Image.open(
                before_file
            ).convert("RGB")

            with col1:

                st.image(
                    before_image,
                    caption="Before",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error loading Before image: {e}"
            )

    if after_file is not None:

        try:

            after_image = Image.open(
                after_file
            ).convert("RGB")

            with col2:

                st.image(
                    after_image,
                    caption="After",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Error loading After image: {e}"
            )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if (
        before_image is not None
        and after_image is not None
    ):

        st.divider()

        if st.button(
            "🔍 Analyze Changes",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing satellite images..."
            ):

                try:

                    analyzer = ChangeAnalyzer()

                    result = analyzer.compare(
                        before_image,
                        after_image
                    )

                    st.success(
                        "✅ Change analysis completed!"
                    )

                    # ------------------------------------------------
                    # METRICS
                    # ------------------------------------------------

                    st.subheader(
                        "📊 Change Analysis Results"
                    )

                    m1, m2, m3 = st.columns(3)

                    with m1:

                        st.metric(
                            "Changed Pixels",
                            f"{result['changed_pixels']:,}"
                        )

                    with m2:

                        st.metric(
                            "Total Pixels",
                            f"{result['total_pixels']:,}"
                        )

                    with m3:

                        st.metric(
                            "Potential Change",
                            f"{result['change_percentage']:.2f}%"
                        )

                    # ------------------------------------------------
                    # BEFORE VS AFTER
                    # ------------------------------------------------

                    st.subheader(
                        "🖼️ Before vs After"
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        st.image(
                            result["before"],
                            caption="Before",
                            use_container_width=True
                        )

                    with c2:

                        st.image(
                            result["after"],
                            caption="After",
                            use_container_width=True
                        )

                    # ------------------------------------------------
                    # DIFFERENCE
                    # ------------------------------------------------

                    st.subheader(
                        "🔍 Pixel Difference"
                    )

                    st.image(
                        result["difference"],
                        caption=(
                            "Pixel-level differences"
                        ),
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # CHANGE MAP
                    # ------------------------------------------------

                    st.subheader(
                        "🗺️ Change Map"
                    )

                    st.image(
                        result["change_overlay"],
                        caption=(
                            "Red areas indicate "
                            "potential changes"
                        ),
                        use_container_width=True
                    )

                    # ------------------------------------------------
                    # BINARY MASK
                    # ------------------------------------------------

                    with st.expander(
                        "View Binary Change Mask"
                    ):

                        st.image(
                            result["change_mask"],
                            caption="Binary Change Mask",
                            use_container_width=True
                        )

                    # ------------------------------------------------
                    # INTERPRETATION
                    # ------------------------------------------------

                    percentage = (
                        result["change_percentage"]
                    )

                    st.subheader(
                        "🧠 Interpretation"
                    )

                    if percentage < 5:

                        st.info(
                            f"Low detected change: "
                            f"{percentage:.2f}% of the image."
                        )

                    elif percentage < 20:

                        st.warning(
                            f"Moderate detected change: "
                            f"{percentage:.2f}% of the image."
                        )

                    else:

                        st.warning(
                            f"High detected change: "
                            f"{percentage:.2f}% of the image."
                        )

                    st.info(
                        """
                        ⚠️ Important: pixel differences do
                        not necessarily represent real-world
                        changes. Lighting, clouds, shadows,
                        seasonal variation and image
                        misalignment may produce differences.
                        """
                    )

                except Exception as e:

                    st.error(
                        "Change analysis failed."
                    )

                    st.exception(e)

    else:

        st.info(
            "Upload both Before and After images "
            "to start change analysis."
        )


# ============================================================
# AI AGENT
# ============================================================

elif page == "🤖 AI Agent":

    st.header("🤖 SatQuery AI Agent")

    st.write(
        """
        SatQuery AI is designed around an agentic
        architecture that routes different remote-sensing
        queries to specialized analysis modules.
        """
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader(
            "🖼️ Single Image VQA"
        )

        st.success("Active")

        st.write(
            "Answers questions related to "
            "uploaded satellite imagery."
        )

    with c2:

        st.subheader(
            "🌍 Scene Captioning"
        )

        st.success("Active")

        st.write(
            "Generates descriptive summaries "
            "of satellite scenes."
        )

    with c3:

        st.subheader(
            "🛰️ Change Analysis"
        )

        st.success("Active")

        st.write(
            "Compares satellite images from "
            "different dates."
        )

    st.divider()

    st.subheader(
        "🏗️ Agent Architecture"
    )

    st.code(
        """
User
 │
 ▼
SatQuery AI Agent
 │
 ├── Single Image VQA
 │
 ├── Scene Captioning
 │
 └── Bi-Temporal Change Analysis
 │
 ▼
Analysis Result
        """,
        language="text"
    )

    st.info(
        """
        Current prototype includes a baseline
        bi-temporal pixel-difference pipeline.
        Vision-language and specialized remote-sensing
        modules can be integrated into the same architecture.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SatQuery AI | Smart India Hackathon 2026 | "
    "Satellite Image Analysis Prototype"
)