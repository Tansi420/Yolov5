from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

st.set_page_config(
    page_title="Inventario de Mudanza Inteligente",
    page_icon="📦",
    layout="wide"
)

# ---------- ESTÉTICA (sin modificar la fuente) ----------
st.markdown("""
<style>
    /* Fondo general con degradado cálido */
    .stApp {
        background: linear-gradient(135deg, #fff7ed 0%, #ffe8d1 50%, #ffd9b3 100%);
    }

    /* Barra lateral estilo cartón */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #8b5e3c 0%, #6b4226 100%);
    }
    [data-testid="stSidebar"] * {
        color: #fff4e6 !important;
    }

    /* Título con degradado */
    h1 {
        background: linear-gradient(90deg, #c2410c, #ea580c, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
        border-bottom: 4px dashed #d97706;
    }

    h2, h3 {
        color: #7c2d12 !important;
    }

    /* Tarjetas para las columnas */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.75);
        border: 2px solid #fdba74;
        border-radius: 18px;
        padding: 1.2rem;
        box-shadow: 0 6px 18px rgba(194, 65, 12, 0.15);
    }

    /* Botones */
    .stButton > button, [data-testid="stCameraInput"] button {
        background: linear-gradient(90deg, #ea580c, #f59e0b);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: bold;
    }
    .stButton > button:hover, [data-testid="stCameraInput"] button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 12px rgba(234, 88, 12, 0.4);
    }

    /* Cámara e imágenes */
    [data-testid="stCameraInput"] {
        border: 3px dashed #ea580c;
        border-radius: 16px;
        padding: 0.5rem;
        background: #fffaf3;
    }
    img {
        border-radius: 14px;
    }

    /* Tabla y mensajes */
    [data-testid="stDataFrame"] {
        border: 2px solid #fdba74;
        border-radius: 12px;
        overflow: hidden;
    }
    .stAlert {
        border-radius: 12px;
        border-left: 6px solid #ea580c;
    }

    hr {
        border-top: 3px dashed #d97706;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title("📦 Inventario de Mudanza Inteligente")
st.markdown(
    "¿Te mudas? Fotografía cada habitación de tu casa y la app identificará y contará "
    "los objetos que hay en ella (sillas, laptops, botellas, libros...) para que "
    "calcules cuántas cajas necesitas y no olvides nada en el camión."
)

with st.spinner("Preparando a tu asistente de empaque..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("🧰 Ajustes del escáner")
        st.subheader("Qué tan estricto es el inventario")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

    picture = st.camera_input("📸 Fotografía una habitación para inventariarla", key="camera")

    if picture:
        bytes_data = picture.getvalue()

        # Decodificar con Pillow en lugar de cv2 (evita dependencia libGL)
        #pil_img  = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        #np_img   = np.array(pil_img)   # array RGB

        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR para que YOLO procese bien

        
        with st.spinner("Contando tus pertenencias..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        result    = results[0]
        boxes     = result.boxes
        annotated = result.plot()              # devuelve BGR numpy array
        annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB sin cv2

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏠 Tu habitación, etiquetada")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("📋 Lista de objetos para empacar")
            if boxes is not None and len(boxes) > 0:
                label_names    = model.names
                category_count = {}
                category_conf  = {}

                for box in boxes:
                    cat  = int(box.cls.item())
                    conf = float(box.conf.item())
                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)

                data = [
                    {
                        "Categoría":          label_names[cat],
                        "Cantidad":           count,
                        "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                    }
                    for cat, count in category_count.items()
                ]

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría")["Cantidad"], color="#ea580c")
            else:
                st.info("No encontramos objetos que empacar con los ajustes actuales.")
                st.caption("Prueba a reducir la confianza mínima en la barra lateral o toma la foto con más luz.")
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Inventario de mudanza con YOLOv5 + Streamlit + PyTorch. ¡Buena suerte con tu nuevo hogar! 🚚")
