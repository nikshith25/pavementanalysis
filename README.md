# Flexible Pavement Analysis Dashboard

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Important model notes

- The reference K exponent in the supplied PDF is unreadable; enter a verified K value.
- Inputs absent from the PDF (vehicle speed, BC/DBM moduli, area mass, fatigue strain and fatigue modulus) must be supplied by the researcher.
- The damping ratio is the PDF's compliance-weighted preliminary approximation.
- The MCK calculation is an equivalent SDOF model using zero initial displacement and velocity.
- Fatigue strain is independent input. The dashboard does not infer tensile strain from DAF.
- The dashboard is a research calculation aid, not a substitute for validation or a layered pavement analysis.
