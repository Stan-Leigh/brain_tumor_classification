# Brain Tumor MRI Classifier

A Streamlit application that classifies uploaded brain MRI images into four categories: glioma tumor, meningioma tumor, pituitary tumor, or no tumor.

> Educational demo only. This project is not a medical diagnostic tool.

## Project files

```text
brain_tumor_app.py
requirements.txt
brain_tumor_classification.ipynb
models/
  brain_tumor_mri_model.keras
  class_names.json
```

## How to train

Open `brain_tumor_classification.ipynb` in Colab or Jupyter, run the cells, and make sure the final cell saves:

```text
models/brain_tumor_mri_model.keras
models/class_names.json
```

## How to run the Streamlit app locally

```bash
pip install -r requirements.txt
streamlit run brain_tumor_app.py
```

## Output

The app returns:

- predicted class
- confidence score
- class probability table
- class probability bar chart
- original image and processed model input image
