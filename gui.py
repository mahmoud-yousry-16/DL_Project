import streamlit as st
from project import (
    analyze_dataset_structure,
    remove_corrupted_images,
    generate_sample_grid
)

st.set_page_config(page_title="Fatigue Dataset Tool", layout="wide")

st.title("Deap Learning Project")

# -------- اختيار مسار الداتا --------
dataset_path = st.text_input("Data")

# -------- 1) تحليل الداتا --------
if st.button("Analyze Dataset Structure"):
    try:
        classes, class_files = analyze_dataset_structure(dataset_path)

        st.success("Data was analysised successfully")
        st.write("## Classes Found:")
        st.write(classes)

        for cls, files in class_files.items():
            st.write(f"### {cls} — {len(files)} image")

    except Exception as e:
        st.error(f"Error: {e}")


# -------- 2) حذف الصور التالفة --------
if st.button("Remove Corrupted Images"):
    try:
        classes, class_files = analyze_dataset_structure(dataset_path)
        cleaned, report = remove_corrupted_images(class_files)

        st.success("Damaged image was deleted")
        st.json(report)

    except Exception as e:
        st.error(f"Error: {e}")


# -------- 3) Sample Grid --------
if st.button("Generate Sample Grid"):
    try:
        classes, class_files = analyze_dataset_structure(dataset_path)
        out_path = generate_sample_grid(class_files)

        st.success("✔ Sample Grid Generated")
        st.image(out_path)

    except Exception as e:
        st.error(f"Error: {e}")