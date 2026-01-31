import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Human–AI Generative Design Studio",
    layout="wide"
)

st.title("🧠 Human–AI Collaborative Generative Design Studio")
st.caption("Domain-aware diffusion system for images & design videos")

# ---------------- DOMAIN OPTIONS ----------------
DOMAIN_OPTIONS = {
    "Architecture": [
        "Interior Design",
        "Exterior Building",
        "Floor Plan Concept",
        "Urban Space",
        "Landscape Architecture"
    ],
    "Engineering": [
        "Lightweight Mechanical Part",
        "Aerospace Component",
        "Automotive Component",
        "Structural Optimization",
        "Industrial Design"
    ],
    "Fashion": [
        "Clothing Design",
        "Fabric Texture",
        "Footwear",
        "Runway Outfit",
        "Accessories"
    ],
    "Molecular Design": [
        "Drug-like Molecule",
        "Protein Binding Concept",
        "Chemical Structure Visualization",
        "Nanostructure"
    ],
    "Virtual Content": [
        "Game Environment",
        "Character Design",
        "Sci-Fi Scene",
        "Fantasy World",
        "Concept Art"
    ]
}

# ---------------- DEVICE ----------------
device = "cpu"
st.sidebar.warning("⚠ Running on CPU (SD-Turbo Fast Mode)")

# ---------------- LOAD MODEL (ONCE) ----------------
@st.cache_resource
def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32,
        safety_checker=None
    )
    pipe.enable_attention_slicing()
    pipe.to(device)
    return pipe

pipe = load_pipe()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎛 Design Controls")

domain = st.sidebar.selectbox("Domain", list(DOMAIN_OPTIONS.keys()))
subcategory = st.sidebar.selectbox("Sub-category", DOMAIN_OPTIONS[domain])

style = st.sidebar.text_input(
    "Style",
    "modern, clean, professional"
)

lighting = st.sidebar.text_input(
    "Mood / Lighting",
    "cinematic lighting"
)

instructions = st.sidebar.text_area(
    "User Instructions",
    "realistic, high clarity, professional visualization"
)

mode = st.sidebar.radio(
    "Generation Mode",
    ["Image", "Video (Frames)"]
)

steps = st.sidebar.slider(
    "Diffusion Steps (Speed vs Quality)",
    2, 4, 2
)

video_seconds = 0
if mode == "Video (Frames)":
    video_seconds = st.sidebar.slider(
        "Video Length (seconds)",
        2, 5, 3
    )

generate = st.sidebar.button("🚀 Generate")

# ---------------- PROMPT ENGINE ----------------
def build_prompt(frame=None):
    motion = ""
    if frame is not None:
        motion = f", cinematic variation, frame {frame}"

    return f"""
    {subcategory} in the domain of {domain},
    style: {style},
    lighting: {lighting},
    {instructions},
    clean composition, high visual clarity{motion}
    """

# ---------------- GENERATION ----------------
if generate:
    st.subheader("🧾 AI-Interpreted Prompt")
    st.code(build_prompt())

    os.makedirs("frames", exist_ok=True)

    if mode == "Image":
        with st.spinner("⚡ Generating image (SD-Turbo)…"):
            img = pipe(
                prompt=build_prompt(),
                num_inference_steps=steps,
                guidance_scale=0.0,
                height=512,
                width=512
            ).images[0]

        st.subheader("🖼 Generated Image")
        st.image(img, use_container_width=True)

    else:
        st.subheader("🎥 Generated Design Frames")
        frames = video_seconds * 3  # ~3 FPS

        cols = st.columns(3)

        for i in range(frames):
            with st.spinner(f"Generating frame {i+1}/{frames}"):
                img = pipe(
                    prompt=build_prompt(i),
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    height=512,
                    width=512
                ).images[0]

                cols[i % 3].image(img, use_container_width=True)

        st.success("✅ Frame sequence generated (concept video)")
