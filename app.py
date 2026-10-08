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

# ---------- ESTÉTICA: etiqueta de envío + cinta de embalaje (sin tocar la fuente) ----------
st.markdown("""
<style>
    :root {
        --tinta: #14213d;
        --cinta: #ffc300;
        --papel: #eaf0f7;
        --sello: #d62828;
        --etiqueta: #ffffff;
    }

    /* Fondo tipo plano de casa: papel cuadriculado */
    .stApp {
        background-color: var(--papel);
        background-image:
            linear-gradient(rgba(20, 33, 61, 0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 33, 61, 0.06) 1px, transparent 1px);
        background-size: 28px 28px;
        color: var(--tinta);
    }

    /* Barra lateral azul tinta con borde de cinta */
    [data-testid="stSidebar"] {
        background: var(--tinta);
        border-right: 8px solid var(--cinta);
    }
    [data-testid="stSidebar"] * {
        color: #f5f7fb !important;
    }
    [data-testid="stSidebar"] h1 {
        background: none;
        box-shadow: none;
        transform: none;
        padding: 0;
        color: var(--cinta) !important;
    }

    /* Título como una tira de cinta pegada, ligeramente inclinada */
    .stApp h1 {
        display: inline-block;
        background: var(--cinta);
        color: var(--tinta);
        padding: 0.35rem 1.2rem;
        transform: rotate(-1deg);
        box-shadow: 4px 4px 0 var(--tinta);
        margin-bottom: 0.8rem;
    }

    h2, h3 {
        color: var(--tinta) !important;
    }

    /* Columnas como etiquetas de envío */
    [data-testid="column"], [data-testid="stColumn"] {
        background: var(--etiqueta);
        border: 2px solid var(--tinta);
        border-left: 14px solid var(--tinta);
        border-radius: 4px;
        padding: 1.2rem 1.4rem;
        box-shadow: 6px 6px 0 rgba(20, 33, 61, 0.18);
    }

    /* Cámara: marco de caja sellada */
    [data-testid="stCameraInput"] {
        border: 3px dashed var(--tinta);
        border-radius: 4px;
        padding: 0.6rem;
        background: var(--etiqueta);
    }
    [data-testid="stCameraInput"] button,
    .stButton > button {
        background: var(--tinta);
        color: var(--cinta);
        border: 2px solid var(--tinta);
        border-radius: 2px;
        font-weight: bold;
    }
    [data-testid="stCameraInput"] button:hover,
    .stButton > button:hover {
        background: var(--cinta);
        color: var(--tinta);
    }

    img {
        border: 2px solid var(--tinta);
        border-radius: 2px;
    }

    /* Métricas como sellos */
    [data-testid="stMetric"] {
        background: var(--papel);
        border: 2px solid var(--sello);
        padding: 0.6rem 0.9rem;
        border-radius: 2px;
    }
    [data-testid="stMetricValue"] {
        color: var(--sello);
    }

    /* Tabla y mensajes */
    [data-testid="stDataFrame"] {
        border: 2px solid var(--tinta);
        border-radius: 2px;
    }
    .stAlert {
        border-radius: 2px;
        border: 2px solid var(--tinta);
        border-left: 10px solid var(--sello);
    }

    /* Divisor: cinta de precaución */
    hr {
        height: 12px;
        border: none;
        background: repeating-linear-gradient(
            -45deg,
            var(--cinta) 0 14px,
            var(--tinta) 14px 28px
        );
        opacity: 1;
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
st.markdown("---")

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

        col1, col2 = st.columns(2, gap="large")

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

                m1, m2 = st.columns(2)
                m1.metric("Objetos totales", len(boxes))
                m2.metric("Tipos de objeto", len(category_count))

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
                st.bar_chart(df.set_index("Categoría")["Cantidad"], color="#14213d")
            else:
                st.info("No encontramos objetos que empacar con los ajustes actuales.")
                st.caption("Prueba a reducir la confianza mínima en la barra lateral o toma la foto con más luz.")
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Inventario de mudanza con YOLOv5 + Streamlit + PyTorch. ¡Buena suerte con tu nuevo hogar! 🚚")    }

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
