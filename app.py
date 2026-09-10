import io
from PIL import Image, ImageOps
from keras.models import load_model
import numpy as np
import streamlit as st

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Registro con Validación Facial",
    page_icon="👤",
    layout="centered",
)


# --- CARGA DEL MODELO Y LABELS ---
@st.cache_resource
def cargar_modelo():
    # Nombre exacto como aparece en tu repositorio de GitHub (m minúscula)
    model = load_model("keras_model.h5", compile=False)

    # Cargar etiquetas desde labels.txt
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    return model, class_names


model, class_names = cargar_modelo()

# --- CABECERA ---
st.title("🔐 Registro de Usuario")
st.write(
    "Para iniciar el registro, primero debemos verificar tu identidad usando la cámara."
)

with st.sidebar:
    st.subheader("Verificación Biométrica")
    st.write(
        "Este sistema valida si eres **Sebastián** antes de habilitar el formulario."
    )

st.divider()

# --- PASO 1: VERIFICACIÓN CON CÁMARA ---
st.subheader("Paso 1: Validación de Identidad")
img_file_buffer = st.camera_input("Toma una foto para validar tu acceso")

es_sebastian = False

if img_file_buffer is not None:
    # --- PROCESAMIENTO OFICIAL DE TEACHABLE MACHINE ---
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(img_file_buffer).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predicción
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # Limpiar nombre de la clase (quita el "0 " o "1 ")
    nombre_clase = class_name.split(" ", 1)[-1] if " " in class_name else class_name

    # --- EVALUACIÓN DE IDENTIDAD ---
    # Si detecta la clase 0 ("sebastian")
    if index == 0 and confidence_score > 0.60:
        st.success(
            f"👤 **Identidad confirmada:** Hola {nombre_clase.capitalize()} ({confidence_score:.0%} de confianza)"
        )
        es_sebastian = True
    else:
        st.error(
            f"🚫 **Acceso denegado:** No se detectó a Sebastián (Registrado como '{nombre_clase}' con {confidence_score:.0%} de certeza)"
        )

# --- PASO 2: FORMULARIO DE REGISTRO CONDICIONAL ---
st.divider()
st.subheader("Paso 2: Datos de Registro")

if es_sebastian:
    with st.form("form_registro"):
        nombre = st.text_input("Nombre Completo", value="Sebastián")
        correo = st.text_input("Correo Electrónico")
        password = st.text_input("Contraseña", type="password")

        submit = st.form_submit_button("Completar Registro")

        if submit:
            if nombre and correo and password:
                st.balloons()
                st.success(
                    f"¡Registro completado con éxito! Bienvenido, {nombre}."
                )
            else:
                st.error("Por favor completa todos los campos del formulario.")
else:
    st.warning(
        "⚠️ Debes validarte en la cámara como **Sebastián** para activar el formulario."
    )
