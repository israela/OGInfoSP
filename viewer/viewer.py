import streamlit as st
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulation Results Viewer", layout="wide")

st.markdown("""<style>
    .block-container { padding-bottom: 1rem; }
    [data-testid="stSidebarUserContent"] { margin-top: -3rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem;
    }
    .st-key-informative_type_options,
    .st-key-informative_type_options [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    .st-key-informative_type_options [data-testid="stRadio"] {
        margin: -0.55rem 0 !important;
    }
    .field-label {
        margin: 0 0 1.7rem;
        font-size: 0.875rem;
        font-weight: 400;
        line-height: 1.6;
    }
    .config-summary {
        margin-top: -0.2rem;
        font-size: 0.875rem;
        line-height: 1.6;
    }
    .config-summary-title {
        margin-bottom: 0.3rem;
        font-weight: 400;
    }
    .config-summary-details {
        display: grid;
        gap: 0.25rem;
        padding-left: 0.65rem;
        color: inherit;
    }
    .config-summary-row {
        display: flex;
        flex-wrap: wrap;
        column-gap: 0.65rem;
    }
    .config-summary-row span:not(:last-child)::after {
        content: '·';
        margin-left: 0.65rem;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-divider) {
        margin: 0.5rem 0 0.7rem;
    }
    .sidebar-divider {
        margin: 0 !important;
        border: 0;
        border-top: 1px solid rgba(128, 128, 128, 0.35);
    }
    .methods-heading {
        margin: 0 0 0.45rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .bulk-vs-heading {
        margin: 0;
        font-size: 0.8rem;
        white-space: nowrap;
    }
    .st-key-bulk_vs_options [data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 0.15rem;
    }
    .st-key-bulk_vs_options [data-testid="stCheckbox"] label {
        align-items: center;
        gap: 0.2rem;
    }
    .st-key-bulk_vs_options [data-testid="stCheckbox"] p {
        font-size: 0.8rem;
        white-space: nowrap;
        transform: translate(-0.4rem, 0.08rem);
    }
</style>""", unsafe_allow_html=True)

# --- Helper functions ---

RESULTS_ROOT = 'output/July_2026'

DATASET_FOLDER_PREFIXES = {
    'Gaussian (K=4)': 'gaussian_K4',
    'Gaussian (K=10)': 'gaussian_K10',
    'CIFAR (3 classes)': 'CIFAR_BCD',
    'CIFAR (10 classes)': 'CIFAR_all10',
}

INFORMATIVE_TYPES = ['exclude_1', 'non_trivial', 'up_to_3', 'single_0_to_6']


def get_cal_split(n_calibration, n_vector_scaling, vector_scaling):
    n_cal = n_calibration if vector_scaling == 'none' else n_calibration - n_vector_scaling
    n_vs = n_vector_scaling if vector_scaling not in ('none', 'none_reduced_cal') else 0
    return n_cal, n_vs


def informative_type_available(dataset_type, informative_type):
    folder_prefix = DATASET_FOLDER_PREFIXES[dataset_type]
    return any(
        name.startswith(f'{folder_prefix}_{informative_type}_')
        for name in os.listdir(RESULTS_ROOT)
    )


def select_informative_type(informative_type):
    st.session_state['informative_type'] = informative_type


def select_vs_for_all(bulk_key, checkbox_keys):
    for key in checkbox_keys:
        st.session_state[key] = st.session_state[bulk_key]


def load_sequences(filepath):
    """Load per-distance (Gaussian) results."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        distances = json.loads(data['training_args']['distances'])
        FCR = json.loads(data['FCR_per_distance'])
        power = json.loads(data['mean_power_per_distance'])
        selected = json.loads(data.get('mean_selected_per_distance', '[]'))
        correct_selected = json.loads(data.get('mean_correct_selected_per_distance', '[]'))
        return distances, FCR, power, selected, correct_selected
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def load_values(filepath):
    """Load distribution (CIFAR) results."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        FCR = json.loads(data['all_FCP'])
        power = json.loads(data['all_power'])
        selected = json.loads(data.get('all_selected', '[]'))
        correct_selected = json.loads(data.get('all_correct_selected', '[]'))
        return FCR, power, selected, correct_selected
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def build_filepath_gaussian(folder, method_name, n_cal, n_vs, size, alpha, iterations, informative_type, vs_label, denom_suffix=''):
    base = f'{folder}/{method_name}_gaussian_cal_{n_cal}_vs_{n_vs}_test_{size}_alpha_{alpha}_iterations_{iterations}_informative_type_{informative_type}'
    if denom_suffix:
        base += f'_{denom_suffix}'
    base += f'_vectorscaling_{vs_label}'
    if 'denominator' in denom_suffix:
        base += '_denom_' + ('increase' if 'cal' in denom_suffix else 'noincrease')
    return base + '.txt'


def build_filepath_cifar(folder, method_name, n_cal, n_vs, size, alpha, iterations, informative_type, vs_label, denom_suffix=''):
    base = f'{folder}/{method_name}_CIFAR10_cal_{n_cal}_vs_{n_vs}_test_{size}_alpha_{alpha}_iterations_{iterations}_informative_type_{informative_type}'
    if denom_suffix:
        base += f'_{denom_suffix}'
    base += f'_vectorscaling_{vs_label}'
    if 'denominator' in denom_suffix:
        base += '_denom_' + ('increase' if 'cal' in denom_suffix else 'noincrease')
    return base + '.txt'


# --- Define all method/VS combinations ---

METHODS_CONFIG = [
    ('OGInfoSP', 'OGInfoSP', '', None),
    ('OGInfoSP-Cal', 'OGInfoSPCal', '', None),
    ('InfoSP', 'InfoSP', '', None),
    ('InfoSCOP', 'InfoSCOP', '', None),
    ('Classic Conformal', 'ClassicConformal', '', None),
    ('Classic Conformal Adaptive Score', 'AdaptiveClassicConformal', '', None),
    ('Top Class', 'TopClass', '', None),
]

VS_OPTIONS = {
    'none': 'No VS',
    'only_bias': 'VS (only bias)',
    'full': 'VS (full)',
    'none_reduced_cal': 'Reduced cal (no VS)',
}

VS_LINESTYLES = {
    'none': '-',
    'only_bias': '--',
    'full': '-.',
    'none_reduced_cal': ':',
}

VS_HATCHES = {'none': '', 'only_bias': '..', 'full': '//', 'none_reduced_cal': ''}

METHOD_COLORS = {
    'OGInfoSP': {'none': '#1f4e8a', 'only_bias': '#4a90d9', 'full': '#3a78c4', 'none_reduced_cal': '#9ecae1'},
    'OGInfoSP-Cal': {'none': '#1a7a3a', 'only_bias': '#4cb86b', 'full': '#33a050', 'none_reduced_cal': '#a1d99b'},
    'InfoSP': {'none': '#b81d1d', 'only_bias': '#e85a5a', 'full': '#d43d3d', 'none_reduced_cal': '#fc9f9f'},
    'InfoSCOP': {'none': '#6a3d9a', 'only_bias': '#9b7ec8', 'full': '#8560b0', 'none_reduced_cal': '#c9b8e8'},
    'Classic Conformal': {'none': '#6b3a1f', 'only_bias': '#a0694e', 'full': '#885038', 'none_reduced_cal': '#d4a888'},
    'Classic Conformal Adaptive Score': {'none': '#0e7c86', 'only_bias': '#3aacb5', 'full': '#25969f', 'none_reduced_cal': '#7fd4db'},
    'Top Class': {'none': '#c4197d', 'only_bias': '#e65aab', 'full': '#d93d95', 'none_reduced_cal': '#f2a0cf'},
}

METRICS = ['FCR', 'Power', 'Selected', 'Correct Selected']


# --- Sidebar ---

st.sidebar.header("📊 Configuration")

dataset_type = st.sidebar.radio("Dataset", ["Gaussian (K=4)", "Gaussian (K=10)", "CIFAR (3 classes)", "CIFAR (10 classes)"])

available_informative_types = {
    informative_type for informative_type in INFORMATIVE_TYPES
    if informative_type_available(dataset_type, informative_type)
}
informative_type = st.session_state.get('informative_type')
if informative_type not in available_informative_types:
    informative_type = next(option for option in INFORMATIVE_TYPES if option in available_informative_types)
    st.session_state['informative_type'] = informative_type

with st.sidebar.container(key='informative_type_options'):
    st.markdown('<div class="field-label">Informative Type</div>', unsafe_allow_html=True)
    for option in INFORMATIVE_TYPES:
        key = f'informative_type_{option}'
        st.session_state[key] = option if option == informative_type else None
        st.radio(
            'Informative Type', [option], index=0 if option == informative_type else None,
            key=key, disabled=option not in available_informative_types,
            label_visibility='collapsed', on_change=select_informative_type, args=(option,),
        )

size = 500

if dataset_type in ("Gaussian (K=4)", "Gaussian (K=10)"):
    subfolder = st.sidebar.selectbox("Train/Test Distribution", [
        'trained_even_test_even',
        'trained_uneven_test_even',
        'trained_even_test_uneven',
        'trained_uneven_test_uneven',
    ])
    alpha = 0.05
    iterations = 10000
    K = 4 if dataset_type == "Gaussian (K=4)" else 10
    folder_base = f'{RESULTS_ROOT}/gaussian_K{K}_{informative_type}_{subfolder}'
    n_calibration = size
    n_vector_scaling = 100
else:
    evenness = st.sidebar.selectbox("Class Distribution", ['even', 'uneven'])
    alpha = 0.1
    iterations = 1000
    n_calibration, n_vector_scaling = 1000, 500
    if dataset_type == "CIFAR (10 classes)":
        folder_base = f'{RESULTS_ROOT}/CIFAR_all10_{informative_type}_{evenness}'
    else:
        folder_base = f'{RESULTS_ROOT}/CIFAR_BCD_{informative_type}_{evenness}'

metric = st.sidebar.selectbox("Metric", METRICS)

st.sidebar.markdown(
    '<hr class="sidebar-divider">', unsafe_allow_html=True,
)
st.sidebar.markdown(
    f'<div class="config-summary"><div class="config-summary-title">For this config:</div>'
    f'<div class="config-summary-details"><div class="config-summary-row">'
    f'<span>Test: {size}</span><span>Cal: {n_calibration}</span><span>VS: {n_vector_scaling}</span></div>'
    f'<div class="config-summary-row"><span>Iterations: {iterations}</span><span>Alpha: {alpha}</span></div></div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
st.sidebar.markdown('<div class="methods-heading">Methods & Vector Scaling</div>', unsafe_allow_html=True)

# Method selection
dataset_name = 'gaussian' if dataset_type in ("Gaussian (K=4)", "Gaussian (K=10)") else 'CIFAR10'
configuration = subfolder if dataset_name == 'gaussian' else evenness
default_vs = 'only_bias' if dataset_name == 'gaussian' else 'full'
method_options = {}
for method_label, method_name, denom_suffix, increase_denom in METHODS_CONFIG:
    for vs_key in VS_OPTIONS:
        n_cal, n_vs = get_cal_split(n_calibration, n_vector_scaling, vs_key)
        vs_label = vs_key.replace('_', '')
        filepath = f'{folder_base}/{method_name}_{dataset_name}_cal_{n_cal}_vs_{n_vs}_test_{size}_alpha_{alpha}_iterations_{iterations}_informative_type_{informative_type}_vectorscaling_{vs_label}.txt'
        available = os.path.isfile(filepath)
        key = f'{dataset_type}_{informative_type}_{configuration}_{method_name}_{vs_key}'
        if key not in st.session_state:
            st.session_state[key] = available and vs_key == default_vs
        elif not available:
            st.session_state[key] = False
        method_options[(method_name, vs_key)] = available, key

bulk_labels = {'none': 'None', 'only_bias': 'Only bias', 'full': 'Full', 'none_reduced_cal': 'Reduced cal'}
with st.sidebar.container(key='bulk_vs_options'):
    bulk_columns = st.columns([1, 1, 1.35, 0.85, 1.7], gap='small')
    with bulk_columns[0]:
        st.markdown('<div class="bulk-vs-heading">VS for all:</div>', unsafe_allow_html=True)
    for index, vs_key in enumerate(VS_OPTIONS):
        checkbox_keys = [
            method_options[(method_name, vs_key)][1]
            for _, method_name, _, _ in METHODS_CONFIG
            if method_options[(method_name, vs_key)][0]
        ]
        bulk_key = f'{dataset_type}_{informative_type}_{configuration}_all_{vs_key}'
        st.session_state[bulk_key] = bool(checkbox_keys) and all(st.session_state[key] for key in checkbox_keys)
        with bulk_columns[index + 1]:
            st.checkbox(
                bulk_labels[vs_key], key=bulk_key, disabled=not checkbox_keys,
                on_change=select_vs_for_all, args=(bulk_key, checkbox_keys),
            )

selected_combos = []
for method_label, method_name, denom_suffix, increase_denom in METHODS_CONFIG:
    with st.sidebar.expander(method_label, expanded=True):
        for vs_key, vs_display in VS_OPTIONS.items():
            available, key = method_options[(method_name, vs_key)]
            if st.checkbox(vs_display, key=key, disabled=not available):
                selected_combos.append((method_label, method_name, denom_suffix, increase_denom, vs_key))


# --- Load and plot data ---

st.title("Simulation Results Viewer")

if not selected_combos:
    st.warning("Select at least one method/VS combination from the sidebar.")
    st.stop()

# Check if folder exists
if not os.path.isdir(folder_base):
    st.warning(f"Output folder not found: `{folder_base}`. Results may not be available for this configuration yet.")
    st.stop()

metric_index = METRICS.index(metric)

if dataset_type in ("Gaussian (K=4)", "Gaussian (K=10)"):
    # Line plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    for method_label, method_name, denom_suffix, increase_denom, vs_key in selected_combos:
        n_cal, n_vs = get_cal_split(n_calibration, n_vector_scaling, vs_key)
        vs_label = vs_key.replace('_', '')
        
        # Build the denom suffix for filename
        if denom_suffix:
            if increase_denom:
                file_denom = f'{denom_suffix}_vectorscaling_{vs_label}_denom_increase'
            else:
                file_denom = f'{denom_suffix}_vectorscaling_{vs_label}_denom_noincrease'
            filepath = f'{folder_base}/{method_name}_gaussian_cal_{n_cal}_vs_{n_vs}_test_{size}_alpha_{alpha}_iterations_{iterations}_informative_type_{informative_type}_{file_denom}.txt'
        else:
            filepath = f'{folder_base}/{method_name}_gaussian_cal_{n_cal}_vs_{n_vs}_test_{size}_alpha_{alpha}_iterations_{iterations}_informative_type_{informative_type}_vectorscaling_{vs_label}.txt'
        
        result = load_sequences(filepath)
        if result is None:
            st.sidebar.warning(f"⚠️ File not found: {os.path.basename(filepath)}")
            continue
        
        distances, fcr, power, selected_data, correct_selected_data = result
        series_list = [fcr, power, selected_data, correct_selected_data]
        series = series_list[metric_index]
        
        if not series:
            continue
        
        vs_suffix = f" ({VS_OPTIONS[vs_key]})" if vs_key != 'none' else ''
        label = f"{method_label}{vs_suffix}"
        color = METHOD_COLORS.get(method_label, {}).get(vs_key, None)
        ax.plot(distances, series, label=label, linestyle=VS_LINESTYLES[vs_key], color=color)
    
    ax.set_xlabel('SNR')
    ax.set_ylabel(metric)
    if metric == 'FCR':
        ax.axhline(y=alpha, color='red', linestyle='--', linewidth=1, label=f'α = {alpha}')
    ax.legend(fontsize='small', loc='best')
    ax.grid(True)
    st.pyplot(fig)
    plt.close()

else:
    # Box plot
    box_data = []
    box_labels = []
    box_combinations = []
    
    for method_label, method_name, denom_suffix, increase_denom, vs_key in selected_combos:
        n_cal, n_vs = get_cal_split(n_calibration, n_vector_scaling, vs_key)
        vs_label = vs_key.replace('_', '')
        
        if denom_suffix:
            if increase_denom:
                file_denom = f'{denom_suffix}_vectorscaling_{vs_label}_denom_increase'
            else:
                file_denom = f'{denom_suffix}_vectorscaling_{vs_label}_denom_noincrease'
            filepath = f'{folder_base}/{method_name}_CIFAR10_cal_{n_cal}_vs_{n_vs}_test_{size}_alpha_{alpha}_iterations_{iterations}_informative_type_{informative_type}_{file_denom}.txt'
        else:
            filepath = f'{folder_base}/{method_name}_CIFAR10_cal_{n_cal}_vs_{n_vs}_test_{size}_alpha_{alpha}_iterations_{iterations}_informative_type_{informative_type}_vectorscaling_{vs_label}.txt'
        
        result = load_values(filepath)
        if result is None:
            st.sidebar.warning(f"⚠️ File not found: {os.path.basename(filepath)}")
            continue
        
        fcr, power, selected_data, correct_selected_data = result
        series_list = [fcr, power, selected_data, correct_selected_data]
        series = series_list[metric_index]
        
        if not series:
            continue
        
        vs_suffix = f"\n({VS_OPTIONS[vs_key]})" if vs_key != 'none' else ''
        label = f"{method_label}{vs_suffix}"
        box_data.append(series)
        box_labels.append(label)
        box_combinations.append((method_label, vs_key))
    
    if box_data:
        fig, ax = plt.subplots(figsize=(max(10, len(box_data) * 1.2), 4.5))
        boxplot = ax.boxplot(
            box_data, tick_labels=box_labels, patch_artist=True, widths=0.62,
            medianprops={'color': '#111111', 'linewidth': 1.2},
            whiskerprops={'color': '#444444'}, capprops={'color': '#444444'},
            flierprops={'marker': '.', 'markersize': 2.5, 'alpha': 0.35},
        )
        for (method_label, vs_key), box in zip(box_combinations, boxplot['boxes']):
            box.set_facecolor(METHOD_COLORS[method_label][vs_key])
            box.set_edgecolor('#333333')
            box.set_hatch(VS_HATCHES[vs_key])
            box.set_alpha(0.78)
        ax.set_ylabel(metric)
        if metric == 'FCR':
            ax.axhline(y=alpha, color='red', linestyle='--', linewidth=0.8, label=f'α = {alpha}')
            ax.legend()
        ax.tick_params(axis='x', labelrotation=25, labelsize=9)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, axis='y', color='#d0d0d0', alpha=0.6, linewidth=0.7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No data found for the selected combinations.")
