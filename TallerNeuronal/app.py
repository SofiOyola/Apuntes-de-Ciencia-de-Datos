import streamlit as st
import math


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Clasificador Cardíaco",
    layout="centered"
)


# =========================================================
# ESTILOS
# Solo usamos HTML aquí para cargar CSS.
# El contenido de la aplicación será Streamlit nativo.
# =========================================================

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background: linear-gradient(
        135deg,
        #fff7fb 0%,
        #f8f2ff 48%,
        #eef8ff 100%
    );
}

/* Contenedor central */
.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Fuente */
html, body, p, div, span, label {
    font-family: "Segoe UI", "Trebuchet MS", sans-serif;
}

/* Título principal */
h1 {
    color: #8b688c !important;
    text-align: center;
    font-weight: 750 !important;
    letter-spacing: -0.5px;
}

/* Subtítulos */
h2, h3 {
    color: #75627f !important;
}

/* Texto */
p {
    color: #655d6c;
}

/* Botón */
div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 16px;

    background: linear-gradient(
        90deg,
        #f2c9dc,
        #dcccf2
    );

    color: #5d5065;
    font-size: 17px;
    font-weight: 700;

    padding: 0.8rem 1rem;

    box-shadow: 0px 5px 16px
        rgba(115, 91, 130, 0.12);

    transition: 0.2s;
}

div.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow: 0px 8px 20px
        rgba(115, 91, 130, 0.18);

    color: #514557;
}

/* Tarjetas métricas */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.75);

    border: 1px solid #eadfee;

    border-radius: 18px;

    padding: 16px;

    box-shadow: 0px 5px 16px
        rgba(115, 91, 130, 0.06);
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.70);

    border: 1px solid #eadfee;

    border-radius: 16px;
}

/* Sliders */
.stSlider label {
    color: #675c70 !important;
    font-weight: 600 !important;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #e6dcea;
}

/* Alertas */
div[data-testid="stAlert"] {
    border-radius: 18px;
}

/* Ocultar elementos innecesarios */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NORMALIZACIÓN
# =========================================================

def normalizar_datos(edad, colesterol):
    """
    Aplica exactamente el mismo preprocesamiento
    usado durante el entrenamiento:

    Z-score * 2
    """

    media_edad = 54.305369
    desviacion_edad = 9.068940

    media_colesterol = 246.926174
    desviacion_colesterol = 51.599203

    x1 = (
        (edad - media_edad)
        / desviacion_edad
    ) * 2

    x2 = (
        (colesterol - media_colesterol)
        / desviacion_colesterol
    ) * 2

    return x1, x2


# =========================================================
# RED NEURONAL FINAL
# Arquitectura: 6 → 4 → 2
# Learning rate: 0.1
# =========================================================

def forward(X1, X2):

    # -----------------------------
    # Capa oculta 1: 6 neuronas
    # -----------------------------

    a1 = max(
        0,
        -0.73
        + (-1.6 * X1)
        + (-2.1 * X2)
    )

    a2 = max(
        0,
        0.46
        + (-0.31 * X1)
        + (-0.65 * X2)
    )

    a3 = max(
        0,
        0.51
        + (-0.73 * X1)
        + (-0.80 * X2)
    )

    a4 = max(
        0,
        -0.56
        + (-0.89 * X1)
        + (-2.8 * X2)
    )

    a5 = max(
        0,
        0.33
        + (0.77 * X1)
        + (0.34 * X2)
    )

    a6 = max(
        0,
        -1.6
        + (0.66 * X1)
        + (2.2 * X2)
    )


    # -----------------------------
    # Capa oculta 2: 4 neuronas
    # -----------------------------

    a7 = max(
        0,
        1.3
        + (-0.25 * a1)
        + (0.19 * a2)
        + (0.72 * a3)
        + (0.14 * a4)
        + (-4.4 * a5)
        + (-1.4 * a6)
    )

    a8 = max(
        0,
        2.8
        + (0.87 * a1)
        + (0.96 * a2)
        + (0.44 * a3)
        + (-1.6 * a4)
        + (-0.39 * a5)
        + (0.55 * a6)
    )

    a9 = max(
        0,
        2.6
        + (-2.0 * a1)
        + (0.52 * a2)
        + (-0.61 * a3)
        + (-0.74 * a4)
        + (-0.43 * a5)
        + (-0.45 * a6)
    )

    a10 = max(
        0,
        1.5
        + (0.22 * a1)
        + (-0.11 * a2)
        + (-1.2 * a3)
        + (2.1 * a4)
        + (0.13 * a5)
        + (-0.43 * a6)
    )


    # -----------------------------
    # Capa oculta 3: 2 neuronas
    # -----------------------------

    a11 = max(
        0,
        1.7
        + (-2.8 * a7)
        + (2.0 * a8)
        + (-1.8 * a9)
        + (-1.5 * a10)
    )

    a12 = max(
        0,
        -3.9
        + (-0.69 * a7)
        + (2.1 * a8)
        + (-1.1 * a9)
        + (0.68 * a10)
    )


    # -----------------------------
    # Salida
    # -----------------------------

    a13 = math.tanh(
        0.080
        + (0.93 * a11)
        + (-0.57 * a12)
    )

    return a13


# =========================================================
# ENCABEZADO
# =========================================================

st.title("Clasificador de Problema Cardíaco")

with st.container(border=True):

    st.markdown(
        """
        Esta aplicación utiliza una **Red Neuronal Artificial**
        entrenada en **ScienxLab Playground** para clasificar un
        registro a partir de dos características:
        **edad** y **nivel de colesterol**.
        """
    )

    st.info(
        "🌷 Ajusta los controles y presiona "
        "**Realizar predicción** para consultar el modelo."
    )


# =========================================================
# DATOS DEL PACIENTE
# =========================================================

st.markdown("### 🌸 Datos del paciente")

st.write(
    "Desliza las barras para seleccionar los valores que deseas evaluar."
)


# -------------------------
# Slider edad
# -------------------------

edad = st.slider(
    "🎂 Edad",
    min_value=29,
    max_value=77,
    value=50,
    step=1,
    help="Selecciona la edad del paciente."
)


# -------------------------
# Slider colesterol
# -------------------------

colesterol = st.slider(
    "🩸 Nivel de colesterol",
    min_value=126,
    max_value=564,
    value=220,
    step=1,
    help="Selecciona el nivel de colesterol."
)


# =========================================================
# RESUMEN VISUAL
# =========================================================

st.write("")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="🎂 Edad seleccionada",
        value=f"{edad} años"
    )

with col2:

    st.metric(
        label="🩸 Colesterol",
        value=f"{colesterol} mg/dL"
    )


st.write("")


# =========================================================
# BOTÓN
# =========================================================

predecir = st.button(
    "✨ Realizar predicción",
    use_container_width=True
)


# =========================================================
# RESULTADO
# =========================================================

if predecir:

    X1, X2 = normalizar_datos(
        edad,
        colesterol
    )

    resultado = forward(
        X1,
        X2
    )

    st.divider()
    st.markdown("### ✨ Resultado de la predicción")

    # =====================================================
    # POSIBLE PROBLEMA CARDÍACO
    # =====================================================

    if resultado >= 0:

        st.error(
            """
            ❤️ **El modelo indica posible problema cardíaco**

            De acuerdo con la edad y el nivel de colesterol ingresados,
            la Red Neuronal Artificial clasificó este registro dentro
            de la categoría asociada a **problema cardíaco**.
            """
        )

        st.caption(
            f"Clase técnica: 1  |  Salida de la red: {resultado:.4f}"
        )

    # =====================================================
    # SIN INDICACIÓN DE PROBLEMA CARDÍACO
    # =====================================================

    else:

        st.success(
            """
            💚 **El modelo no indica problema cardíaco**

            De acuerdo con la edad y el nivel de colesterol ingresados,
            la Red Neuronal Artificial clasificó este registro dentro
            de la categoría asociada a **ausencia de problema cardíaco**.
            """
        )

        st.caption(
            f"Clase técnica: -1  |  Salida de la red: {resultado:.4f}"
        )

    # =====================================================
    # ADVERTENCIA
    # =====================================================

    st.warning(
        """
        ⚠️ **Importante:** este resultado corresponde únicamente a un
        modelo académico entrenado con las variables edad y colesterol.
        No constituye un diagnóstico médico ni reemplaza una valoración
        realizada por un profesional de la salud.
        """
    )

    # =====================================================
    # DETALLES TÉCNICOS
    # =====================================================

    with st.expander("🔎 Ver detalles técnicos de la predicción"):

        detalle1, detalle2 = st.columns(2)

        with detalle1:
            st.metric(
                "Edad normalizada",
                f"{X1:.4f}"
            )

        with detalle2:
            st.metric(
                "Colesterol normalizado",
                f"{X2:.4f}"
            )

        st.metric(
            "Salida de la red neuronal",
            f"{resultado:.4f}"
        )

        st.write(
            """
            La salida utiliza la función **tanh**, por lo que el
            resultado se encuentra entre **-1 y 1**.
            """
        )

        st.write(
            """
            - Salida menor que 0 → **Clase -1: sin indicación de problema cardíaco**
            - Salida igual o mayor que 0 → **Clase 1: posible problema cardíaco**
            """
        )

# =========================================================
# INFORMACIÓN DEL MODELO
# =========================================================

st.divider()

st.markdown("### 🧠 Información del modelo")

with st.container(border=True):

    modelo1, modelo2 = st.columns(2)

    with modelo1:

        st.markdown(
            """
            **Arquitectura**

            `6 → 4 → 2`

            **Learning rate**

            `0.1`

            **Activación**

            `ReLU`
            """
        )

    with modelo2:

        st.markdown(
            """
            **Función de pérdida**

            `Squared Error`

            **Training Loss**

            `0.224`

            **Test Loss**

            `0.213`
            """
        )


# =========================================================
# EXPLICACIÓN DEL PREPROCESAMIENTO
# =========================================================

with st.expander(
    "🌷 ¿Cómo se procesan los datos?"
):

    st.write(
        """
        Antes de ingresar a la red neuronal,
        la edad y el colesterol se transforman
        utilizando exactamente el mismo procedimiento
        aplicado durante el entrenamiento.
        """
    )

    st.markdown("#### Edad")

    st.latex(
        r"""
        X_1 =
        2
        \left(
        \frac{
        Edad - 54.305369
        }{
        9.068940
        }
        \right)
        """
    )

    st.markdown("#### Colesterol")

    st.latex(
        r"""
        X_2 =
        2
        \left(
        \frac{
        Colesterol - 246.926174
        }{
        51.599203
        }
        \right)
        """
    )

    st.write(
        """
        De esta manera, la aplicación utiliza
        la misma escala de datos con la que fue
        entrenada la Red Neuronal Artificial.
        """
    )


# =========================================================
# PIE DE PÁGINA
# =========================================================

st.divider()

st.caption(
    "💗 Proyecto académico desarrollado con Streamlit "
    "a partir de una Red Neuronal Artificial entrenada "
    "en ScienxLab Playground."
)