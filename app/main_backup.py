import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np

from ai_model import SatelliteAI
from caption_model import SatelliteCaptioner
from land_cover import LandCoverAnalyzer


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_ai_models():
    vqa_model = SatelliteAI()
    caption_model = SatelliteCaptioner()
    land_cover_model = LandCoverAnalyzer()

    return vqa_model, caption_model, land_cover_model


class ChangeAnalyzer:

    def compare(self, before_image, after_image):

        before = before_image.convert("RGB")
        after = after_image.convert("RGB")

        width = min(before.width, after.width)
        height = min(before.height, after.height)

        before = before.resize((width, height))
        after = after.resize((width, height))

        before_array = np.asarray(before).astype(np.int16)
        after_array = np.asarray(after).astype(np.int16)

        difference_array = np.abs(before_array - after_array)

        difference_gray = difference_array.mean(axis=2)

        threshold = 30

        change_mask_array = (
            difference_gray > threshold
        ).astype(np.uint8) * 255

        changed_pixels = int(
            np.count_nonzero(change_mask_array)
        )

        total_pixels = width * height

        change_percentage = (
            changed_pixels / total_pixels * 100
            if total_pixels > 0
            else 0
        )

        difference_display = np.clip(
            difference_array,
            0,
            255
        ).astype(np.uint8)

        difference_image = Image.fromarray(
            difference_display
        )

        difference_image = ImageEnhance.Contrast(
            difference_image
        ).enhance(3.0)

        change_mask = Image.fromarray(
            change_mask_array
        )

        after_array_uint8 = np.asarray(after).copy()

        after_array_uint8[
            change_mask_array > 0
        ] = [255, 0, 0]

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


# Header

st.title("🛰️ SatQuery AI")

st.subheader(
    "Interactive Vision-Language Assistant for Satellite Image Analysis"
)

st.write(
    "Analyze satellite imagery using AI-assisted image "
    "understanding and bi-temporal change detection."
)


# Sidebar

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


# Home

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
        """
        This is a prototype. Pixel differences can be caused by
        lighting, clouds, shadows, seasonal changes, or image
        misalignment.
        """
    )


# Single Image Analysis

elif page == "📡 Single Image Analysis":

    st.header("📡 Satellite Image Analysis")

    st.write(
        "Upload a satellite image to perform AI-based analysis."
    )

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

            # Load AI models

            with st.spinner(
                "Loading AI models... This may take a while the first time."
            ):

                try:

                    vqa_model, caption_model, land_cover_model = (
                        load_ai_models()
                    )

                    st.success(
                        "AI models loaded successfully!"
                    )

                except Exception as e:

                    st.error(
                        "Unable to load AI models."
                    )

                    st.exception(e)

                    st.stop()


            # Scene Captioning

            st.subheader("🌍 Scene Captioning")

            if st.button(
                "📝 Generate Scene Caption",
                use_container_width=True
            ):

                with st.spinner(
                    "Generating scene description..."
                ):

                    try:

                        caption = (
                            caption_model.generate_caption(
                                image
                            )
                        )

                        st.success(
                            "Scene description generated!"
                        )

                        st.write(
                            f"**AI Description:** {caption}"
                        )

                    except Exception as e:

                        st.error(
                            "Caption generation failed."
                        )

                        st.exception(e)


            st.divider()


            # VQA

            st.subheader(
                "🤖 Visual Question Answering"
            )

            question = st.text_input(
                "Ask a question about the image",
                placeholder=(
                    "Example: What can you see in this image?"
                )
            )

            if st.button(
                "🔍 Ask AI",
                type="primary",
                use_container_width=True
            ):

                if question.strip():

                    with st.spinner(
                        "AI is analyzing the image..."
                    ):

                        try:

                            answer = (
                                vqa_model.answer_question(
                                    image,
                                    question
                                )
                            )

                            st.success(
                                "Answer generated!"
                            )

                            st.write(
                                f"**Question:** {question}"
                            )

                            st.info(
                                f"**AI Answer:** {answer}"
                            )

                        except Exception as e:

                            st.error(
                                "Question answering failed."
                            )

                            st.exception(e)

                else:

                    st.warning(
                        "Please enter a question."
                    )


            st.divider()


            # Land Cover

            st.subheader(
                "🌱 Land-Cover Analysis"
            )

            if st.button(
                "🌍 Analyze Land Cover",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing possible land-cover types..."
                ):

                    try:

                        detected = (
                            land_cover_model.analyze(
                                image
                            )
                        )

                        if detected:

                            st.success(
                                "Possible land-cover categories detected:"
                            )

                            for category in detected:

                                st.write(
                                    f"• **{category}**"
                                )

                        else:

                            st.warning(
                                "No land-cover category was detected."
                            )

                        st.caption(
                            "Note: This is a prototype "
                            "BLIP-based land-cover analysis."
                        )

                    except Exception as e:

                        st.error(
                            "Land-cover analysis failed."
                        )

                        st.exception(e)


        except Exception as e:

            st.error(
                f"Unable to process image: {e}"
            )


# Change Analysis

elif page == "🛰️ Change Analysis":

    st.header(
        "🛰️ Bi-Temporal Change Analysis"
    )

    st.write(
        """
        Upload two images of the same area from different
        dates to visualize potential changes.
        """
    )

    col1, col2 = st.columns(2)

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


                    st.subheader(
                        "🔍 Pixel Difference"
                    )

                    st.image(
                        result["difference"],
                        caption="Pixel-level differences",
                        use_container_width=True
                    )


                    st.subheader(
                        "🗺️ Change Map"
                    )

                    st.image(
                        result["change_overlay"],
                        caption=(
                            "Red areas indicate potential changes"
                        ),
                        use_container_width=True
                    )


                    with st.expander(
                        "View Binary Change Mask"
                    ):

                        st.image(
                            result["change_mask"],
                            caption="Binary Change Mask",
                            use_container_width=True
                        )


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
                        ⚠️ Important: pixel differences do not
                        necessarily represent real-world changes.
                        Lighting, clouds, shadows, seasonal variation
                        and image misalignment may produce differences.
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


# AI Agent

elif page == "🤖 AI Agent":

    st.header(
        "🤖 SatQuery AI Agent"
    )

    st.write(
        """
        SatQuery AI is designed around an agentic architecture
        that routes different remote-sensing queries to specialized
        analysis modules.
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
            "Answers questions related to uploaded satellite imagery."
        )


    with c2:

        st.subheader(
            "🌐 Scene Captioning"
        )

        st.success("Active")

        st.write(
            "Generates descriptive summaries of satellite scenes."
        )


    with c3:

        st.subheader(
            "🛰️ Change Analysis"
        )

        st.success("Active")

        st.write(
            "Compares satellite images from different dates."
        )


    st.divider()


    st.subheader(
        "🌱 Land-Cover Analysis"
    )

    st.success("Active")

    st.write(
        """
        Performs prototype land-cover analysis using
        BLIP-based vision-language prompting.
        """
    )


    st.divider()


    st.subheader(
        "🗺️ Agent Architecture"
    )

    st.code(
        """
User
 ↓
SatQuery AI Agent
 ↓
 ├── Single Image VQA
 ├── Scene Captioning
 ├── Land-Cover Analysis
 └── Bi-Temporal Change Analysis
 ↓
Analysis Result
""",
        language="text"
    )


    st.info(
        """
        The current prototype combines BLIP-based
        vision-language analysis with a baseline
        bi-temporal pixel-difference pipeline.
        """
    )


st.divider()

st.caption(
    "SatQuery AI | Smart India Hackathon 2026 | "
    "Satellite Image Analysis Prototype"
)