import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime
import cv2
import os

# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="PPE Compliance Monitor",
    page_icon="🦺",
    layout="wide"
)

# ─── Constants ─────────────────────────────────────────────
FINAL_CLASSES = {
    0: 'hardhat',
    1: 'no_hardhat',
    2: 'safety_vest',
    3: 'no_safety_vest',
    4: 'person',
    5: 'fall_detected'
}

MODEL_PATH = r"D:\PGDBA\Personal Projects\DL-Project\PPE  project\DL-PPE Project\outputs\best_model.pth"
DEVICE     = torch.device('cpu')

# ─── Load PPE classifier ───────────────────────────────────
@st.cache_resource
def load_ppe_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 6)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

# ─── Load person detector ──────────────────────────────────
@st.cache_resource
def load_detector():
    detector = models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT'
    )
    detector.eval()
    return detector

# ─── Load head detector ────────────────────────────────────
@st.cache_resource
def load_head_detector():
    cascade_path = cv2.data.haarcascades + \
                   'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

# ─── Transforms ────────────────────────────────────────────
ppe_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─── Detect persons ────────────────────────────────────────
def detect_persons(detector, image, threshold=0.6):
    img_tensor = transforms.ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = detector(img_tensor)[0]
    boxes = []
    for score, label, box in zip(
        outputs['scores'], outputs['labels'], outputs['boxes']
    ):
        if label == 1 and score > threshold:
            x1, y1, x2, y2 = box.int().tolist()
            boxes.append((x1, y1, x2, y2))
    return boxes

# ─── Detect head region ────────────────────────────────────
def detect_head_region(head_detector, image_np,
                        person_box, img_w, img_h):
    x1, y1, x2, y2 = person_box
    person_h        = y2 - y1
    center_x        = (x1 + x2) // 2
    head_w          = (x2 - x1) // 2
    person_crop_np  = image_np[y1:y2, x1:x2]

    if person_crop_np.size == 0:
        return (
            max(0, center_x - head_w // 2),
            y1,
            min(img_w, center_x + head_w // 2),
            y1 + int(person_h * 0.25)
        )

    gray  = cv2.cvtColor(person_crop_np, cv2.COLOR_RGB2GRAY)
    faces = head_detector.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=3, minSize=(20, 20)
    )

    if len(faces) > 0:
        fx, fy, fw, fh = sorted(faces, key=lambda f: f[1])[0]
        helmet_y1 = max(0, fy - fh)
        helmet_y2 = fy + fh
        return (
            x1 + fx,
            y1 + helmet_y1,
            x1 + fx + fw,
            y1 + helmet_y2
        )
    else:
        return (
            max(0, center_x - head_w // 2),
            y1,
            min(img_w, center_x + head_w // 2),
            y1 + int(person_h * 0.25)
        )

# ─── Grad-CAM ──────────────────────────────────────────────
def get_gradcam(model, image_pil, target_class):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import (
            ClassifierOutputTarget
        )
        target_layers = [model.features[-1]]
        img_resized   = image_pil.resize((224, 224))
        img_np        = np.array(img_resized).astype(
            np.float32) / 255.0
        img_tensor    = ppe_transform(img_resized).unsqueeze(0)

        with GradCAM(model=model, target_layers=target_layers) as cam:
            targets   = [ClassifierOutputTarget(target_class)]
            grayscale = cam(
                input_tensor=img_tensor, targets=targets
            )
            cam_image = show_cam_on_image(
                img_np, grayscale[0], use_rgb=True
            )
        return cam_image
    except Exception:
        return None

# ─── Classify PPE per worker ──────────────────────────────
def classify_ppe(ppe_model, head_detector, image,
                 person_boxes, ppe_threshold=0.3):
    results  = []
    W, H     = image.size
    image_np = np.array(image)

    for (x1, y1, x2, y2) in person_boxes:
        pad      = 10
        x1p      = max(0, x1 - pad)
        y1p      = max(0, y1 - pad)
        x2p      = min(W, x2 + pad)
        y2p      = min(H, y2 + pad)
        person_h = y2p - y1p

        hx1, hy1, hx2, hy2 = detect_head_region(
            head_detector, image_np,
            (x1p, y1p, x2p, y2p), W, H
        )
        head_crop = image.crop((hx1, hy1, hx2, hy2))

        vest_crop = image.crop((
            x1p,
            y1p + person_h // 3,
            x2p,
            y1p + 2 * person_h // 3
        ))

        full_crop = image.crop((x1p, y1p, x2p, y2p))

        # ── Helmet prediction ──────────────────────────────
        helmet_pred    = 'unknown'
        helmet_conf    = 0.0
        helmet_gradcam = None

        if head_crop.size[0] > 10 and head_crop.size[1] > 10:
            with torch.no_grad():
                tensor      = ppe_transform(
                    head_crop).unsqueeze(0)
                out         = ppe_model(tensor)
                helmet_prob = torch.softmax(out, dim=1)[0]

            hardhat_prob   = helmet_prob[0].item()
            no_helmet_prob = helmet_prob[1].item()

            if hardhat_prob >= ppe_threshold:
                helmet_pred    = 'hardhat'
                helmet_conf    = hardhat_prob
                helmet_gradcam = get_gradcam(
                    ppe_model, head_crop, 0
                )
            elif no_helmet_prob >= ppe_threshold:
                helmet_pred    = 'no_hardhat'
                helmet_conf    = no_helmet_prob
                helmet_gradcam = get_gradcam(
                    ppe_model, head_crop, 1
                )
            else:
                helmet_pred = 'unknown'
                helmet_conf = max(hardhat_prob, no_helmet_prob)

        # ── Vest prediction ────────────────────────────────
        vest_pred    = 'unknown'
        vest_conf    = 0.0
        vest_gradcam = None

        if vest_crop.size[0] > 10 and vest_crop.size[1] > 10:
            with torch.no_grad():
                tensor    = ppe_transform(vest_crop).unsqueeze(0)
                out       = ppe_model(tensor)
                vest_prob = torch.softmax(out, dim=1)[0]

            safety_vest_prob    = vest_prob[2].item()
            no_safety_vest_prob = vest_prob[3].item()

            if safety_vest_prob >= ppe_threshold:
                vest_pred    = 'safety_vest'
                vest_conf    = safety_vest_prob
                vest_gradcam = get_gradcam(
                    ppe_model, vest_crop, 2
                )
            elif no_safety_vest_prob >= ppe_threshold:
                vest_pred    = 'no_safety_vest'
                vest_conf    = no_safety_vest_prob
                vest_gradcam = get_gradcam(
                    ppe_model, vest_crop, 3
                )
            else:
                vest_pred = 'unknown'
                vest_conf = max(
                    safety_vest_prob, no_safety_vest_prob
                )

        # ── Fall detection ─────────────────────────────────
        fall_prob = 0.0
        if full_crop.size[0] > 10 and full_crop.size[1] > 10:
            with torch.no_grad():
                tensor    = ppe_transform(full_crop).unsqueeze(0)
                out       = ppe_model(tensor)
                probs     = torch.softmax(out, dim=1)
                fall_prob = probs[0][5].item()

        results.append({
            'box'           : (x1, y1, x2, y2),
            'helmet'        : helmet_pred,
            'helmet_conf'   : helmet_conf,
            'vest'          : vest_pred,
            'vest_conf'     : vest_conf,
            'fall_prob'     : fall_prob,
            'fall'          : fall_prob > 0.7,
            'helmet_gradcam': helmet_gradcam,
            'vest_gradcam'  : vest_gradcam,
            'head_crop'     : head_crop,
            'vest_crop'     : vest_crop
        })

    return results

# ─── Draw results with confidence on box ──────────────────
def draw_results(image, results):
    annotated = image.copy()
    draw      = ImageDraw.Draw(annotated)

    for r in results:
        x1, y1, x2, y2 = r['box']
        helmet_ok = r['helmet'] == 'hardhat'
        vest_ok   = r['vest']   == 'safety_vest'
        is_fall   = r['fall']
        unknown   = (r['helmet'] == 'unknown'
                     and r['vest'] == 'unknown')

        if is_fall:
            color = 'orange'
            label = 'FALL DETECTED'
        elif unknown:
            color = 'gray'
            label = 'Low confidence'
        elif helmet_ok and vest_ok:
            color = 'green'
            label = 'Compliant'
        elif not helmet_ok and not vest_ok:
            color = 'red'
            label = 'No Helmet + No Vest'
        elif not helmet_ok:
            color = 'red'
            label = 'No Helmet'
        else:
            color = 'red'
            label = 'No Vest'

        # Add confidence to box label
        h_conf     = int(r['helmet_conf'] * 100)
        v_conf     = int(r['vest_conf'] * 100)
        conf_label = f"{label} | H:{h_conf}% V:{v_conf}%"

        draw.rectangle(
            [x1, y1, x2, y2], outline=color, width=3
        )
        label_w = len(conf_label) * 7 + 4
        draw.rectangle(
            [x1, y1 - 22, x1 + label_w, y1], fill=color
        )
        draw.text(
            (x1 + 2, y1 - 20), str(conf_label), fill='white'
        )

    return annotated

# ─── Generate report ───────────────────────────────────────
def generate_report(results, image_name):
    total     = len(results)
    helmet_v  = sum(
        1 for r in results if r['helmet'] == 'no_hardhat'
    )
    vest_v    = sum(
        1 for r in results if r['vest'] == 'no_safety_vest'
    )
    falls     = sum(1 for r in results if r['fall'])
    compliant = sum(
        1 for r in results
        if r['helmet'] == 'hardhat'
        and r['vest']  == 'safety_vest'
        and not r['fall']
    )

    compliance_rate = round(
        (compliant / total * 100) if total > 0 else 0, 1
    )

    if falls > 0:
        risk       = 'CRITICAL'
        risk_color = 'red'
    elif compliance_rate >= 90:
        risk       = 'LOW'
        risk_color = 'green'
    elif compliance_rate >= 70:
        risk       = 'MEDIUM'
        risk_color = 'orange'
    else:
        risk       = 'HIGH'
        risk_color = 'red'

    recommendations = []
    if falls > 0:
        recommendations.append(
            f"EMERGENCY: {falls} potential fall event(s) detected. "
            f"Dispatch safety officer immediately."
        )
    if helmet_v > 0:
        recommendations.append(
            f"{helmet_v} worker(s) detected without helmets. "
            f"Issue stop-work notice until compliance achieved."
        )
    if vest_v > 0:
        recommendations.append(
            f"{vest_v} worker(s) detected without safety vests. "
            f"Ensure high-visibility vests worn at all times."
        )
    if not recommendations:
        recommendations.append(
            "All detected workers are fully compliant. "
            "Continue routine monitoring."
        )

    return {
        'date'             : datetime.now().strftime(
            '%d %B %Y %H:%M'
        ),
        'image'            : image_name,
        'total_workers'    : total,
        'compliant'        : compliant,
        'helmet_violations': helmet_v,
        'vest_violations'  : vest_v,
        'falls'            : falls,
        'compliance_rate'  : compliance_rate,
        'risk'             : risk,
        'risk_color'       : risk_color,
        'recommendations'  : recommendations
    }

# ─── Streamlit UI ──────────────────────────────────────────
st.title("🦺 PPE Compliance Monitoring System")
st.markdown(
    "**Construction Site Safety — "
    "AI Powered Compliance Detection**"
)
st.markdown("---")

with st.spinner('Loading PPE classifier...'):
    ppe_model = load_ppe_model()
with st.spinner('Loading person detector...'):
    detector = load_detector()
with st.spinner('Loading head detector...'):
    head_detector = load_head_detector()

st.success('All models loaded successfully')

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.title("System Info")
st.sidebar.markdown("""
**PPE Classifier** : EfficientNet-B0
**Person Detector**: Faster RCNN ResNet50
**Head Detector**  : OpenCV Haar Cascade
**Test Accuracy**  : 96.16%
**Dataset**        : 32,862 images
**Classes**        : 6 PPE categories
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Legend**")
st.sidebar.markdown("""
🟢 Green  = Fully compliant
🔴 Red    = Violation detected
🟠 Orange = Fall detected
⚫ Gray   = Low confidence
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Detection Settings**")
person_threshold = st.sidebar.slider(
    "Person Detection Confidence",
    min_value=0.3, max_value=0.9,
    value=0.6, step=0.05,
    help="Lower = detects more people."
)
ppe_threshold = st.sidebar.slider(
    "PPE Classification Confidence",
    min_value=0.1, max_value=0.9,
    value=0.3, step=0.05,
    help="Lower = flags more violations."
)
show_gradcam = st.sidebar.checkbox(
    "Show Grad-CAM Explainability",
    value=False,
    help="Show heatmaps explaining model decisions per worker"
)

# ─── Session state for compliance trend ───────────────────
if 'compliance_history' not in st.session_state:
    st.session_state.compliance_history = []
if 'image_history' not in st.session_state:
    st.session_state.image_history = []

# ─── Sample images ─────────────────────────────────────────
st.subheader("Upload Construction Site Image")

SAMPLE_DIR = (
    r"D:\PGDBA\Personal Projects\DL-Project\PPE  project"
    r"\DL-PPE Project\data\raw\test\images"
)

sample_files    = []
selected_sample = None

if os.path.exists(SAMPLE_DIR):
    all_samples  = [
        f for f in os.listdir(SAMPLE_DIR)
        if f.endswith('.jpg')
    ]
    sample_files = (
        all_samples[:4]
        if len(all_samples) >= 4
        else all_samples
    )

if sample_files:
    st.markdown("**Or click a sample image to test:**")
    sample_cols = st.columns(len(sample_files))
    for i, (col, fname) in enumerate(
        zip(sample_cols, sample_files)
    ):
        fpath = os.path.join(SAMPLE_DIR, fname)
        try:
            thumb = Image.open(fpath).convert('RGB').resize(
                (150, 150)
            )
            col.image(thumb)
            if col.button(
                f"Use Sample {i+1}", key=f"sample_{i}"
            ):
                selected_sample = fpath
        except Exception:
            pass

# ─── Batch upload ──────────────────────────────────────────
upload_mode = st.radio(
    "Upload mode",
    ["Single image", "Batch — multiple images"],
    horizontal=True
)

uploaded_files  = []
uploaded_file   = None
image_to_process = None
image_name       = None

if upload_mode == "Single image":
    uploaded_file = st.file_uploader(
        "Upload JPG or PNG image",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file:
        image_to_process = Image.open(
            uploaded_file).convert('RGB')
        image_name = uploaded_file.name
    elif selected_sample:
        image_to_process = Image.open(
            selected_sample).convert('RGB')
        image_name = os.path.basename(selected_sample)

else:
    uploaded_files = st.file_uploader(
        "Upload multiple JPG or PNG images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )

# ─── Process single image ──────────────────────────────────
def process_single(image, image_name):
    image = image.resize((640, 640))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, width=640)

    with st.spinner('Detecting workers...'):
        person_boxes = detect_persons(
            detector, image, threshold=person_threshold
        )

    st.info(f"Detected {len(person_boxes)} worker(s) in image")

    if len(person_boxes) == 0:
        st.warning(
            "No workers detected. Try lowering the "
            "Person Detection Confidence slider."
        )
        return None

    with st.spinner('Analyzing PPE compliance...'):
        results = classify_ppe(
            ppe_model, head_detector, image,
            person_boxes, ppe_threshold
        )

    annotated = draw_results(image, results)
    with col2:
        st.subheader("Compliance Detection")
        st.image(annotated, width=640)

    report = generate_report(results, image_name)

    st.markdown("---")
    st.subheader("📋 Safety Compliance Report")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Workers Detected",  report['total_workers'])
    m2.metric("Fully Compliant",   report['compliant'])
    m3.metric("Helmet Violations", report['helmet_violations'])
    m4.metric("Vest Violations",   report['vest_violations'])
    m5.metric("Compliance Rate",
              f"{report['compliance_rate']}%")

    # Color coded risk level
    if report['risk'] == 'CRITICAL':
        st.error(
            "🚨 Risk Level: CRITICAL — "
            "Immediate action required"
        )
    elif report['risk'] == 'HIGH':
        st.error(
            "⚠️ Risk Level: HIGH — "
            "Multiple violations detected"
        )
    elif report['risk'] == 'MEDIUM':
        st.warning(
            "⚠️ Risk Level: MEDIUM — "
            "Some violations detected"
        )
    else:
        st.success(
            "✅ Risk Level: LOW — "
            "Site is largely compliant"
        )

    st.markdown("#### Per Worker Analysis")
    for i, r in enumerate(results, 1):
        helmet_icon = (
            "✅" if r['helmet'] == 'hardhat'
            else ("❓" if r['helmet'] == 'unknown' else "❌")
        )
        vest_icon = (
            "✅" if r['vest'] == 'safety_vest'
            else ("❓" if r['vest'] == 'unknown' else "❌")
        )
        fall_icon = "🚨 FALL DETECTED" if r['fall'] else ""
        conf_text = (
            f"Helmet conf: {r['helmet_conf']:.0%} | "
            f"Vest conf: {r['vest_conf']:.0%}"
        )
        st.markdown(
            f"**Worker {i}:** "
            f"{helmet_icon} Helmet ({r['helmet']}) | "
            f"{vest_icon} Vest ({r['vest']}) "
            f"{fall_icon}  *{conf_text}*"
        )

    if show_gradcam:
        st.markdown("---")
        st.subheader("🔍 Grad-CAM Explainability")
        st.markdown(
            "Red/yellow = high model attention. "
            "Blue = low attention."
        )
        for i, r in enumerate(results, 1):
            st.markdown(f"**Worker {i}**")
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.markdown("Head crop")
            gc1.image(
                r['head_crop'].resize((112, 112)), width=112
            )
            if r['helmet_gradcam'] is not None:
                gc2.markdown(f"Helmet CAM ({r['helmet']})")
                gc2.image(r['helmet_gradcam'], width=112)
            gc3.markdown("Vest crop")
            gc3.image(
                r['vest_crop'].resize((112, 112)), width=112
            )
            if r['vest_gradcam'] is not None:
                gc4.markdown(f"Vest CAM ({r['vest']})")
                gc4.image(r['vest_gradcam'], width=112)

    report_text = f"""
SITE SAFETY COMPLIANCE REPORT
==============================
Date             : {report['date']}
Image            : {report['image']}
Workers Detected : {report['total_workers']}
Fully Compliant  : {report['compliant']}
Helmet Violations: {report['helmet_violations']}
Vest Violations  : {report['vest_violations']}
Falls Detected   : {report['falls']}
Compliance Rate  : {report['compliance_rate']}%
Risk Level       : {report['risk']}

RECOMMENDATIONS:
"""
    for i, rec in enumerate(report['recommendations'], 1):
        report_text += f"{i}. {rec}\n"

    st.text(report_text)
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=(
            f"safety_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        ),
        mime="text/plain"
    )

    return report

# ─── Main flow ─────────────────────────────────────────────
if upload_mode == "Single image" and image_to_process:
    report = process_single(image_to_process, image_name)
    if report:
        st.session_state.compliance_history.append(
            report['compliance_rate']
        )
        st.session_state.image_history.append(
            image_name[:20]
        )

elif upload_mode == "Batch — multiple images" \
        and len(uploaded_files) > 0:

    st.subheader(
        f"Processing {len(uploaded_files)} images..."
    )

    batch_reports    = []
    compliance_rates = []

    for idx, uf in enumerate(uploaded_files):
        st.markdown(f"---")
        st.markdown(f"### Image {idx+1}: {uf.name[:40]}")
        img = Image.open(uf).convert('RGB')
        r   = process_single(img, uf.name)
        if r:
            batch_reports.append(r)
            compliance_rates.append(r['compliance_rate'])

    if batch_reports:
        st.markdown("---")
        st.subheader("📊 Batch Summary Report")

        avg_compliance = round(
            sum(compliance_rates) / len(compliance_rates), 1
        )
        total_workers  = sum(
            r['total_workers'] for r in batch_reports
        )
        total_helmet_v = sum(
            r['helmet_violations'] for r in batch_reports
        )
        total_vest_v   = sum(
            r['vest_violations'] for r in batch_reports
        )
        total_falls    = sum(
            r['falls'] for r in batch_reports
        )

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Images Processed", len(batch_reports))
        b2.metric("Total Workers",    total_workers)
        b3.metric("Helmet Violations",total_helmet_v)
        b4.metric("Vest Violations",  total_vest_v)
        b5.metric("Avg Compliance",   f"{avg_compliance}%")

        if total_falls > 0:
            st.error(
                f"🚨 {total_falls} fall event(s) detected "
                f"across batch. Immediate investigation required."
            )

        # Compliance chart across batch
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        colors  = [
            'green' if r >= 90
            else ('orange' if r >= 70 else 'red')
            for r in compliance_rates
        ]
        ax.bar(
            range(1, len(compliance_rates) + 1),
            compliance_rates,
            color=colors
        )
        ax.axhline(
            y=90, color='green',
            linestyle='--', label='Target 90%'
        )
        ax.axhline(
            y=70, color='orange',
            linestyle='--', label='Warning 70%'
        )
        ax.set_xlabel('Image Number')
        ax.set_ylabel('Compliance Rate (%)')
        ax.set_title('Compliance Rate Across Batch')
        ax.set_ylim(0, 100)
        ax.legend()
        st.pyplot(fig)
        plt.close()

        # Consolidated batch report download
        batch_text = f"""
BATCH SAFETY COMPLIANCE REPORT
================================
Date              : {datetime.now().strftime('%d %B %Y %H:%M')}
Images Processed  : {len(batch_reports)}
Total Workers     : {total_workers}
Avg Compliance    : {avg_compliance}%
Helmet Violations : {total_helmet_v}
Vest Violations   : {total_vest_v}
Falls Detected    : {total_falls}

PER IMAGE SUMMARY:
"""
        for i, r in enumerate(batch_reports, 1):
            batch_text += (
                f"\nImage {i}: {r['image'][:40]}\n"
                f"  Workers: {r['total_workers']} | "
                f"Compliance: {r['compliance_rate']}% | "
                f"Risk: {r['risk']}\n"
            )

        st.download_button(
            label="📥 Download Batch Report",
            data=batch_text,
            file_name=(
                f"batch_report_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            ),
            mime="text/plain"
        )

# ─── Compliance trend chart ────────────────────────────────
if len(st.session_state.compliance_history) > 1:
    st.markdown("---")
    st.subheader("📈 Compliance Trend")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(
        range(1, len(
            st.session_state.compliance_history) + 1
        ),
        st.session_state.compliance_history,
        'b-o', markersize=6
    )
    ax.axhline(
        y=90, color='green',
        linestyle='--', label='Target 90%'
    )
    ax.axhline(
        y=70, color='orange',
        linestyle='--', label='Warning 70%'
    )
    ax.fill_between(
        range(1, len(
            st.session_state.compliance_history) + 1
        ),
        st.session_state.compliance_history,
        alpha=0.1, color='blue'
    )
    ax.set_xlabel('Upload Number')
    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('Session Compliance Trend')
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)
    plt.close()

    if st.button("Clear History"):
        st.session_state.compliance_history = []
        st.session_state.image_history      = []
        st.rerun()